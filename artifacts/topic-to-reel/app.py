import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import io
import time
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
import bar_chart_race as bcr

# ── OpenAI client (Replit AI Integrations) ─────────────────────────────────────
client = OpenAI(
    base_url=os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL"),
    api_key=os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY"),
)

# ── Helpers ────────────────────────────────────────────────────────────────────

MUSIC_URLS = [
    "https://cdn.pixabay.com/download/audio/2021/09/23/audio_57bc8dcb4e.mp3",
    "https://cdn.pixabay.com/download/audio/2022/10/25/audio_d0bba6d4e2.mp3",
    "https://cdn.pixabay.com/download/audio/2022/03/15/audio_c8f4f4a5db.mp3",
]


def extract_data_from_llm(topic: str) -> tuple[pd.DataFrame, str]:
    """Call LLM to produce a wide-format CSV suitable for bar chart race."""
    prompt = f"""You are a precise data research assistant.

Generate a structured dataset for a bar chart race animation about:
"{topic}"

Requirements:
- First column: time index (e.g. Year as integer, e.g. 1990, 1995, 2000 …)
- 5-10 category columns with short names (≤20 chars each)
- Numeric values only — no commas inside numbers, no units, no currency symbols
- At least 15 time periods for smooth animation
- Values must change meaningfully over time
- No missing values

Return ONLY the raw CSV text — no markdown fences, no explanation, nothing else.

Example:
Year,USA,China,Japan,Germany,UK
1990,5963,390,3132,1547,1090
1995,7664,734,5449,2591,1342
"""
    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": "Return only clean CSV data. No markdown. No explanations."},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=4000,
    )

    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences
    lines = raw.splitlines()
    cleaned = "\n".join(l for l in lines if not l.strip().startswith("```"))

    df = pd.read_csv(io.StringIO(cleaned))
    index_col = df.columns[0]
    df = df.set_index(index_col)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Derive a clean chart title from the topic
    title_resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "user", "content": f"Write a short chart title (max 8 words, title case) for: {topic}. Return only the title."},
        ],
        max_completion_tokens=40,
    )
    chart_title = title_resp.choices[0].message.content.strip().strip('"').strip("'")
    return df, chart_title


def dark_fig(figsize):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#000000")
    ax.set_facecolor("#000000")
    return fig, ax


def create_race_video(df: pd.DataFrame, raw_path: str) -> str:
    """Render the bar chart race to an MP4 file (9:16, 1080×1920)."""
    # 1080 / 160 = 6.75  |  1920 / 160 = 12.0
    figsize = (6.75, 12.0)
    dpi = 160

    fig, ax = dark_fig(figsize)
    plt.close(fig)

    bcr.bar_chart_race(
        df=df,
        filename=raw_path,
        orientation="h",
        sort="desc",
        n_bars=min(10, len(df.columns)),
        fixed_order=False,
        fixed_max=True,
        steps_per_period=25,
        interpolate_period=False,
        label_bars=True,
        bar_size=0.82,
        period_label={
            "x": 0.97,
            "y": 0.12,
            "ha": "right",
            "va": "center",
            "size": 52,
            "color": "white",
            "fontweight": "bold",
            "fontfamily": "DejaVu Sans",
        },
        period_length=600,
        figsize=figsize,
        dpi=dpi,
        cmap="dark12",
        title="",
        title_size=0,
        bar_label_size=13,
        tick_label_size=12,
        shared_fontdict={"family": "DejaVu Sans", "color": "white"},
        scale="linear",
        writer="ffmpeg",
        bar_kwargs={"alpha": 0.88, "edgecolor": "none"},
        filter_column_colors=False,
    )
    return raw_path


def make_title_frame(title: str, width: int, height: int) -> np.ndarray:
    """Render a semi-transparent title banner as an RGBA numpy array."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Banner background
    banner_h = max(90, int(height * 0.07))
    draw.rectangle([(0, 0), (width, banner_h)], fill=(0, 0, 0, 200))

    # Try to load a bold font; fall back to default
    font_size = max(32, int(banner_h * 0.45))
    try:
        font = ImageFont.truetype("/run/current-system/sw/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    # Wrap text if too long
    max_chars = max(20, width // (font_size // 2))
    if len(title) > max_chars:
        title = title[: max_chars - 3] + "…"

    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (width - tw) // 2
    ty = (banner_h - th) // 2
    draw.text((tx, ty), title, fill=(255, 255, 255, 240), font=font)

    return np.array(img)


def download_music(tmp_dir: str):
    for url in MUSIC_URLS:
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200 and len(r.content) > 10_000:
                path = os.path.join(tmp_dir, "music.mp3")
                with open(path, "wb") as f:
                    f.write(r.content)
                return path
        except Exception:
            continue
    return None


def post_produce(raw_path: str, title: str, final_path: str, tmp_dir: str) -> str:
    """Add title overlay + background music; export final 1080×1920 MP4."""
    from moviepy.editor import (
        VideoFileClip,
        ImageClip,
        CompositeVideoClip,
        AudioFileClip,
        concatenate_audioclips,
    )

    video = VideoFileClip(raw_path)
    duration = video.duration
    w, h = video.size

    # Title overlay
    title_arr = make_title_frame(title, w, h)
    title_clip = (
        ImageClip(title_arr, ismask=False)
        .set_duration(duration)
        .set_position(("center", "top"))
    )

    composite = CompositeVideoClip([video, title_clip])

    # Background music
    music_path = download_music(tmp_dir)
    if music_path:
        try:
            audio = AudioFileClip(music_path)
            if audio.duration < duration:
                loops = int(np.ceil(duration / audio.duration))
                audio = concatenate_audioclips([audio] * loops)
            audio = audio.subclip(0, duration).volumex(0.35)
            composite = composite.set_audio(audio)
        except Exception:
            pass

    composite.write_videofile(
        final_path,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=os.path.join(tmp_dir, "tmp_audio.m4a"),
        remove_temp=True,
        logger=None,
        preset="fast",
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )

    video.close()
    composite.close()
    return final_path


# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Topic-to-Reel", page_icon="🎬", layout="centered")

st.title("Topic-to-Reel")
st.caption("Enter a topic — get a 1080×1920 bar chart race Reel ready to post.")

topic = st.text_area(
    "Topic or raw data",
    placeholder=(
        "e.g.  Top 10 Fastest Aircraft Mach Speeds over time\n"
        "      Global CO₂ emissions by country 1960-2020\n"
        "      Or paste a CSV directly"
    ),
    height=110,
)

generate = st.button("Generate Reel", type="primary", use_container_width=True)

if generate:
    if not topic.strip():
        st.warning("Please enter a topic or paste some data first.")
        st.stop()

    progress = st.progress(0, text="Starting…")
    status = st.empty()

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_mp4 = os.path.join(tmp_dir, "race.mp4")
            final_mp4 = os.path.join(tmp_dir, "reel.mp4")

            # ── Stage 1: Research / Parse Data ────────────────────────────────
            progress.progress(8, text="Researching data…")
            status.info("🔍  Researching and structuring data with AI…")

            df, chart_title = extract_data_from_llm(topic)

            progress.progress(33, text="Data ready — building animation…")
            status.info(f"📊  Generating animation for: **{chart_title}**")

            # ── Stage 2: Generate Animation ───────────────────────────────────
            create_race_video(df, raw_mp4)

            progress.progress(68, text="Animation done — finalizing Reel…")
            status.info("🎬  Adding title overlay and music…")

            # ── Stage 3: Post-Production ──────────────────────────────────────
            post_produce(raw_mp4, chart_title, final_mp4, tmp_dir)

            progress.progress(100, text="Done!")
            status.success("✅  Your Reel is ready!")

            with open(final_mp4, "rb") as f:
                video_bytes = f.read()

        st.video(video_bytes)

        st.download_button(
            label="⬇️  Download MP4 (1080×1920)",
            data=video_bytes,
            file_name=f"reel_{int(time.time())}.mp4",
            mime="video/mp4",
            use_container_width=True,
        )

    except Exception as exc:
        progress.empty()
        status.empty()
        st.error(f"Something went wrong: {exc}")
        with st.expander("Error details"):
            st.exception(exc)
