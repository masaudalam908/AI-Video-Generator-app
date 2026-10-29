import os
import re
import json
import math
import tempfile
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    AudioFileClip, ImageClip, VideoFileClip, CompositeVideoClip,
    concatenate_videoclips, CompositeAudioClip
)
import edge_tts
import asyncio
from gtts import gTTS

# ---------------------------- App configuration ----------------------------
st.set_page_config(page_title="AI Video Generator", page_icon="🎬", layout="wide")
st.title("🎬 AI Video Generator")
st.caption("Turn a topic into a narrated MP4 video with visuals, subtitles, and background music.")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
PEXELS_API_URL = "https://api.pexels.com/v1/videos/search"
VIDEO_W, VIDEO_H = 1280, 720
FPS = 24


def get_secret(name: str, default: str = "") -> str:
    """Read a key from Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.getenv(name, default).strip()


def request_gemini_script(topic: str, duration_minutes: int, api_key: str) -> dict:
    """Generate a narration script and scene plan using Gemini REST API."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={quote(api_key)}"
    )
    # Rough narration pace: 125 words/minute. Keep scenes short enough to be editable.
    target_words = duration_minutes * 125
    prompt = f"""
Create a factual, engaging educational video plan about: {topic!r}.
Target runtime: {duration_minutes} minutes. Narration target: about {target_words} words.
Return ONLY valid JSON with this exact structure:
{{
  "title": "short video title",
  "scenes": [
    {{
      "narration": "spoken narration for this scene, roughly 70-110 words",
      "visual_query": "2-5 concrete English keywords for finding relevant stock footage",
      "caption": "short subtitle-style scene heading"
    }}
  ]
}}
Create enough scenes to reach the target word count (approximately {duration_minutes * 125} words overall).
Use about 1-2 scenes per minute, with roughly 60-100 words per scene; prioritize coherent narration and distinct visual queries.
Use accurate, accessible language. Do not include markdown fences. Avoid unsupported claims and fabricated citations.
"""
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"temperature": 0.6, "responseMimeType": "application/json"}},
        timeout=180,
    )
    if response.status_code >= 400:
        detail = response.text[:1200]
        raise RuntimeError(f"Gemini API error ({response.status_code}): {detail}")
    payload = response.json()
    try:
        raw = payload["candidates"][0]["content"]["parts"][0]["text"]
        data = json.loads(raw)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("Gemini returned an unexpected response. Try again with a simpler topic.") from exc
    scenes = data.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise RuntimeError("Gemini did not return any scenes. Please try again.")
    cleaned = []
    for item in scenes:
        if not isinstance(item, dict):
            continue
        narration = str(item.get("narration", "")).strip()
        if narration:
            cleaned.append({
                "narration": narration,
                "visual_query": str(item.get("visual_query", topic)).strip() or topic,
                "caption": str(item.get("caption", "")).strip(),
            })
    if not cleaned:
        raise RuntimeError("The generated script was empty. Please try again.")
    data["scenes"] = cleaned
    data["title"] = str(data.get("title", topic)).strip() or topic
    return data


def fetch_pexels_video(query: str, api_key: str, used_ids: set):
    """Return a downloaded landscape stock clip, or None if unavailable."""
    try:
        response = requests.get(
            PEXELS_API_URL,
            headers={"Authorization": api_key},
            params={"query": query, "orientation": "landscape", "per_page": 8},
            timeout=35,
        )
        if response.status_code != 200:
            return None
        videos = response.json().get("videos", [])
        # Prefer landscape MP4s around 720p, skipping clips already used when possible.
        candidates = [v for v in videos if v.get("id") not in used_ids] or videos
        for video in candidates:
            files = sorted(
                [f for f in video.get("video_files", [])
                 if f.get("file_type") == "video/mp4" and f.get("width") and f.get("height")],
                key=lambda f: abs((f.get("width", 0) / max(f.get("height", 1), 1)) - (16 / 9))
                         + abs(f.get("height", 0) - 720) / 2000,
            )
            if not files:
                continue
            try:
                r = requests.get(files[0]["link"], timeout=60, stream=True)
                if r.status_code == 200:
                    fd, path = tempfile.mkstemp(suffix=".mp4")
                    os.close(fd)
                    with open(path, "wb") as out:
                        for chunk in r.iter_content(chunk_size=1024 * 512):
                            if chunk:
                                out.write(chunk)
                    if os.path.getsize(path) > 20_000:
                        used_ids.add(video.get("id"))
                        return path
                    os.remove(path)
            except requests.RequestException:
                continue
    except requests.RequestException:
        return None
    return None


def make_placeholder(query: str, path: str):
    """Create a clean title card when stock footage cannot be fetched."""
    img = Image.new("RGB", (VIDEO_W, VIDEO_H), (20, 31, 52))
    draw = ImageDraw.Draw(img)
    try:
        font_big = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 25)
    except OSError:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()
    title = query[:65]
    draw.rounded_rectangle((75, 130, VIDEO_W - 75, VIDEO_H - 130), radius=28, fill=(32, 52, 82))
    draw.text((110, 275), "VISUAL: " + title, font=font_big, fill="white")
    draw.text((110, 365), "Stock footage unavailable for this scene", font=font_small, fill=(210, 225, 245))
    img.save(path, quality=92)


async def make_edge_voice(text: str, output_path: str, voice: str):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def create_voice(text: str, output_path: str, voice: str):
    """Try Edge TTS; use gTTS as a fallback if Edge TTS fails."""
    try:
        asyncio.run(make_edge_voice(text, output_path, voice))
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return
    except Exception:
        pass
    try:
        gTTS(text=text, lang="en", slow=False).save(output_path)
    except Exception as exc:
        raise RuntimeError(
            "Could not create voice-over. Check internet access or try again later."
        ) from exc


def fit_video(path: str, duration: float):
    clip = VideoFileClip(path, audio=False)
    if clip.duration <= 0:
        clip.close()
        raise RuntimeError("A downloaded stock clip was invalid.")
    if clip.duration < duration:
        # Loop short stock clips to fill the scene duration.
        loops = max(1, math.ceil(duration / clip.duration))
        pieces = [clip.copy() for _ in range(loops)]
        video = concatenate_videoclips(pieces, method="compose").subclip(0, duration)
        clip.close()
    else:
        video = clip.subclip(0, duration)
    # Center-crop to 16:9, then resize.
    target_ratio = VIDEO_W / VIDEO_H
    if video.w / video.h > target_ratio:
        new_w = int(video.h * target_ratio)
        x1 = max(0, (video.w - new_w) // 2)
        video = video.crop(x1=x1, y1=0, x2=x1 + new_w, y2=video.h)
    else:
        new_h = int(video.w / target_ratio)
        y1 = max(0, (video.h - new_h) // 2)
        video = video.crop(x1=0, y1=y1, x2=video.w, y2=y1 + new_h)
    return video.resize((VIDEO_W, VIDEO_H))


def make_srt(scenes, durations, output_path):
    def stamp(seconds):
        ms = int(round(seconds * 1000))
        hours, ms = divmod(ms, 3_600_000)
        minutes, ms = divmod(ms, 60_000)
        secs, ms = divmod(ms, 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"
    cursor = 0.0
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (scene, duration) in enumerate(zip(scenes, durations), 1):
            text = scene["narration"].strip().replace("\n", " ")
            f.write(f"{i}\n{stamp(cursor)} --> {stamp(cursor + duration)}\n{text}\n\n")
            cursor += duration


def generate_video(topic, minutes, gemini_key, pexels_key, voice, progress, uploaded_music=None):
    workdir = tempfile.mkdtemp(prefix="ai_video_")
    try:
        progress.progress(5, text="Writing the script…")
        plan = request_gemini_script(topic, minutes, gemini_key)
        scenes = plan["scenes"]

        # Voice is rendered scene-by-scene so each scene can be aligned to a visual.
        audio_paths, voice_durations = [], []
        for i, scene in enumerate(scenes):
            progress.progress(8 + int(27 * (i + 1) / len(scenes)),
                              text=f"Creating voice-over ({i + 1}/{len(scenes)})…")
            audio_path = os.path.join(workdir, f"voice_{i:03}.mp3")
            create_voice(scene["narration"], audio_path, voice)
            audio_paths.append(audio_path)
            a = AudioFileClip(audio_path)
            voice_durations.append(max(a.duration, 1.0))
            a.close()

        # Scale scene durations to match the requested runtime while preserving narration.
        target_seconds = minutes * 60
        total_voice = sum(voice_durations)
        # Use the spoken audio duration; target runtime is approximate, not padded with silence.
        # The generated narration aims for ~125 words/minute; warn if it differs significantly.
        if abs(total_voice - target_seconds) / target_seconds > 0.25:
            st.warning(
                f"Generated narration is about {total_voice / 60:.1f} minutes, "
                f"rather than {minutes} minutes. Runtime depends on the voice service and script length."
            )

        progress.progress(40, text="Finding relevant stock footage…")
        used_ids = set()
        scene_clips = []
        for i, (scene, audio_path, duration) in enumerate(zip(scenes, audio_paths, voice_durations)):
            progress.progress(40 + int(28 * (i + 1) / len(scenes)),
                              text=f"Preparing visuals ({i + 1}/{len(scenes)})…")
            visual_path = fetch_pexels_video(scene["visual_query"], pexels_key, used_ids) if pexels_key else None
            if visual_path:
                try:
                    visual = fit_video(visual_path, duration)
                    os.remove(visual_path)
                except Exception:
                    try:
                        os.remove(visual_path)
                    except OSError:
                        pass
                    visual_path = None
            if not visual_path:
                image_path = os.path.join(workdir, f"visual_{i:03}.jpg")
                make_placeholder(scene["visual_query"], image_path)
                visual = ImageClip(image_path).set_duration(duration)
            voice_clip = AudioFileClip(audio_path)
            visual = visual.set_audio(voice_clip)
            scene_clips.append(visual)

        progress.progress(72, text="Assembling video and subtitles…")
        final = concatenate_videoclips(scene_clips, method="compose")
        # Optional user-supplied background music. Use music you own or are licensed to use.
        music_path = None
        if uploaded_music is not None:
            music_path = os.path.join(workdir, "background_music" + Path(uploaded_music.name).suffix)
            with open(music_path, "wb") as mf:
                mf.write(uploaded_music.getvalue())
        if music_path:
            music = AudioFileClip(music_path)
            music = music.audio_loop(duration=final.duration).volumex(0.10)
            final_audio = CompositeAudioClip([final.audio, music])
            final = final.set_audio(final_audio)

        output_path = os.path.join(workdir, "generated_video.mp4")
        final.write_videofile(
            output_path, fps=FPS, codec="libx264", audio_codec="aac",
            preset="ultrafast", threads=2, logger=None,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        srt_path = os.path.join(workdir, "subtitles.srt")
        make_srt(scenes, voice_durations, srt_path)
        progress.progress(100, text="Video ready!")
        return output_path, srt_path, plan["title"], workdir
    except Exception:
        shutil.rmtree(workdir, ignore_errors=True)
        raise


with st.sidebar:
    st.header("Settings")
    duration = st.slider("Video duration (minutes)", min_value=3, max_value=15, value=3)
    voice = st.selectbox("AI voice", [
        "en-US-AriaNeural", "en-US-GuyNeural", "en-GB-SoniaNeural", "en-GB-RyanNeural"
    ])
    st.caption("Voice availability depends on the text-to-speech provider and internet access.")

topic = st.text_area("Enter your video topic", placeholder="Example: How solar panels generate electricity", height=100)
music_upload = st.file_uploader("Optional background music (MP3/WAV/M4A; use audio you have rights to)", type=["mp3", "wav", "m4a", "aac", "ogg"])
st.info("A Gemini API key is required. A Pexels API key improves the video with stock footage. Without Pexels, the app uses generated title cards. Background music is optional.")
if st.button("✨ Generate Video", type="primary", use_container_width=True):
    if not topic.strip():
        st.error("Please enter a topic.")
        st.stop()
    gemini_key = get_secret("GEMINI_API_KEY")
    pexels_key = get_secret("PEXELS_API_KEY")
    if not gemini_key:
        st.error("Missing GEMINI_API_KEY. Add it in Streamlit Community Cloud → App → Settings → Secrets, or set it as an environment variable locally.")
        st.stop()
    progress = st.progress(0, text="Starting…")
    try:
        with st.spinner("Generating your video. Longer videos may take several minutes."):
            output_path, srt_path, title, workdir = generate_video(
                topic.strip(), duration, gemini_key, pexels_key, voice, progress, music_upload
            )
        st.success(f"Created: {title}")
        st.video(output_path)
        with open(output_path, "rb") as video_file:
            st.download_button(
                "⬇️ Download Video (MP4)", data=video_file,
                file_name=re.sub(r"[^a-zA-Z0-9_-]+", "_", title).strip("_")[:80] + ".mp4",
                mime="video/mp4", use_container_width=True
            )
        with open(srt_path, "rb") as subtitle_file:
            st.download_button(
                "Download Subtitles (SRT)", data=subtitle_file,
                file_name="subtitles.srt", mime="application/x-subrip"
            )
        st.caption("Stock footage provided by Pexels where available. Generated files live temporarily on the server and may disappear when the app restarts.")
        st.markdown("Stock footage via [Pexels](https://www.pexels.com/).")
    except Exception as exc:
        st.error(f"Video generation failed: {exc}")
        st.caption("Check that API keys are valid, the app has internet access, and the selected service quotas have not been exceeded.")
