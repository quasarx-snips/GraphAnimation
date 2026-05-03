import shutil
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
_ffmpeg_path = shutil.which("ffmpeg")
if _ffmpeg_path:
    matplotlib.rcParams["animation.ffmpeg_path"] = _ffmpeg_path
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation
import matplotlib.ticker as mticker
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

# ── OpenAI client ──────────────────────────────────────────────────────────────
client = OpenAI(
    base_url=os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL"),
    api_key=os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY"),
)

# ── Constants ──────────────────────────────────────────────────────────────────
FONT_PATH = "/run/current-system/sw/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG   = "/run/current-system/sw/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Vibrant, high-contrast palette for black background
LINE_COLORS = [
    "#4FC3F7",  # sky blue
    "#FF7043",  # deep orange
    "#66BB6A",  # green
    "#FFA726",  # amber
    "#CE93D8",  # lavender
    "#26C6DA",  # cyan
    "#F06292",  # pink
    "#A5D6A7",  # mint
]

MUSIC_URLS = [
    "https://cdn.pixabay.com/download/audio/2021/09/23/audio_57bc8dcb4e.mp3",
    "https://cdn.pixabay.com/download/audio/2022/10/25/audio_d0bba6d4e2.mp3",
    "https://cdn.pixabay.com/download/audio/2022/03/15/audio_c8f4f4a5db.mp3",
]

BG = "#000000"
FPS = 30
STEPS_PER_PERIOD = 30   # interpolation frames between each data point


# ── Helpers ────────────────────────────────────────────────────────────────────

def _try_font(size: int):
    for path in (FONT_PATH, FONT_REG):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def smart_fmt(val: float) -> str:
    """Format numbers cleanly: B / M suffixes or plain integer."""
    if abs(val) >= 1_000_000_000:
        return f"{val/1_000_000_000:.1f}B"
    if abs(val) >= 1_000_000:
        return f"{val/1_000_000:.1f}M"
    if abs(val) >= 10_000:
        return f"{val:,.0f}"
    if abs(val) >= 100:
        return f"{val:.0f}"
    return f"{val:.2f}"


# ── Data extraction ────────────────────────────────────────────────────────────

def extract_data_from_llm(topic: str) -> tuple[pd.DataFrame, str]:
    prompt = f"""You are a data research assistant producing data for an animated line chart.

Topic: "{topic}"

Rules:
- First column: numeric time index (Year or date as integer, e.g. 2000, 2001, …)
- 2 to 6 category columns, short names (≤18 chars)
- Numeric values only — no commas inside numbers, no units, no symbols
- Provide ONE ROW PER YEAR (or per logical time step) — do NOT skip years
- Minimum 15 rows, ideally 20-30 rows
- Values must change realistically over time
- No missing values

Return ONLY raw CSV — no markdown, no explanation, no code fences.

Example format:
Year,Male,Female
2000,3041,2974
2001,3065,2999
"""
    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": "Return only clean CSV. No markdown. No explanations."},
            {"role": "user",   "content": prompt},
        ],
        max_completion_tokens=4000,
    )
    raw = resp.choices[0].message.content.strip()
    cleaned = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("```"))
    df = pd.read_csv(io.StringIO(cleaned))
    df = df.set_index(df.columns[0])
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.ffill().bfill().fillna(0)

    title_resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content":
            f"Write a short chart title (max 7 words, title case) for: {topic}. Return only the title."}],
        max_completion_tokens=30,
    )
    chart_title = title_resp.choices[0].message.content.strip().strip('"\'')
    return df, chart_title


# ── Line race animation (pure matplotlib FuncAnimation) ───────────────────────

def create_line_race_video(df: pd.DataFrame, chart_title: str, raw_path: str) -> str:
    """Render growing-line race to a 9:16 MP4 (1080×1920)."""

    n_periods = len(df)
    n_lines   = min(len(df.columns), len(LINE_COLORS))
    df        = df.iloc[:, :n_lines]
    cols      = list(df.columns)
    colors    = LINE_COLORS[:n_lines]

    # ── Interpolate between data points for smooth motion ────────────────────
    total_frames = (n_periods - 1) * STEPS_PER_PERIOD + 1

    x_raw = np.arange(n_periods, dtype=float)
    x_interp = np.linspace(0, n_periods - 1, total_frames)

    y_interp = {}
    for col in cols:
        y_interp[col] = np.interp(x_interp, x_raw, df[col].values)

    # x-axis labels: map integer indices back to original index values
    idx_labels = [str(v) for v in df.index]

    # ── Figure setup ──────────────────────────────────────────────────────────
    FIG_W, FIG_H = 6.75, 12.0   # 1080×1920 @ 160 DPI
    DPI = 160

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG, dpi=DPI)

    # Layout: top title area, main chart, bottom padding
    gs = fig.add_gridspec(
        nrows=3, ncols=1,
        height_ratios=[0.10, 0.78, 0.12],
        left=0.14, right=0.82,
        top=0.96, bottom=0.06,
        hspace=0,
    )
    title_ax = fig.add_subplot(gs[0])
    ax       = fig.add_subplot(gs[1])
    _        = fig.add_subplot(gs[2])

    # ── Title area ────────────────────────────────────────────────────────────
    title_ax.set_facecolor(BG)
    title_ax.axis("off")
    title_ax.text(
        0.5, 0.55, chart_title,
        transform=title_ax.transAxes,
        color="white", fontsize=18, fontweight="bold",
        ha="center", va="center", wrap=True,
        fontfamily="DejaVu Sans",
    )

    # ── Chart area styling ────────────────────────────────────────────────────
    ax.set_facecolor(BG)
    ax.tick_params(colors="#888888", labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#333333")
        ax.spines[spine].set_linewidth(0.8)

    # Subtle horizontal grid
    ax.yaxis.grid(True, color="#1e1e1e", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)

    # Axes limits
    all_y = np.concatenate([y_interp[c] for c in cols])
    y_min, y_max = all_y.min(), all_y.max()
    y_pad = (y_max - y_min) * 0.08
    ax.set_xlim(-0.05, n_periods - 0.95)
    ax.set_ylim(y_min - y_pad, y_max + y_pad * 2.5)

    # X-axis ticks — show ~6 evenly spaced labels
    tick_step = max(1, n_periods // 6)
    tick_pos   = list(range(0, n_periods, tick_step))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([idx_labels[i] for i in tick_pos], color="#888888", fontsize=9)

    # Y-axis formatter
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: smart_fmt(v)))
    ax.tick_params(axis="y", labelcolor="#888888", labelsize=9)

    # ── Per-series artists ────────────────────────────────────────────────────
    main_lines, glow_lines, dots, val_labels = [], [], [], []

    for i, col in enumerate(cols):
        c = colors[i]
        glow, = ax.plot([], [], color=c, linewidth=10, alpha=0.15, solid_capstyle="round", zorder=2)
        main, = ax.plot([], [], color=c, linewidth=2.5,  solid_capstyle="round", zorder=3)
        dot,  = ax.plot([], [], "o", color=c, markersize=9, zorder=5)
        lbl   = ax.text(0, 0, "", color=c, fontsize=11, fontweight="bold",
                        va="center", ha="left", zorder=6, fontfamily="DejaVu Sans")
        glow_lines.append(glow)
        main_lines.append(main)
        dots.append(dot)
        val_labels.append(lbl)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        plt.Line2D([0], [0], color=colors[i], linewidth=2.5, label=cols[i])
        for i in range(n_lines)
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        framealpha=0.15,
        facecolor="#111111",
        edgecolor="#333333",
        labelcolor="white",
        fontsize=10,
        handlelength=1.8,
        borderpad=0.6,
        labelspacing=0.5,
    )

    # ── Year / period counter ─────────────────────────────────────────────────
    period_txt = ax.text(
        0.98, 0.06, "",
        transform=ax.transAxes,
        color="#ffffff", fontsize=36, fontweight="bold",
        ha="right", va="bottom", alpha=0.85,
        fontfamily="DejaVu Sans",
    )

    # ── Source label ──────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.015, "randomdatavstime",
        ha="center", va="bottom", fontsize=8,
        color="#444444", fontfamily="DejaVu Sans",
    )

    # ── Animation function ────────────────────────────────────────────────────
    def update(frame: int):
        f = min(frame, total_frames - 1)
        x_now = x_interp[:f + 1]

        for i, col in enumerate(cols):
            y_now = y_interp[col][:f + 1]
            main_lines[i].set_data(x_now, y_now)
            glow_lines[i].set_data(x_now, y_now)

            if len(x_now) > 0:
                xend, yend = x_now[-1], y_now[-1]
                dots[i].set_data([xend], [yend])
                val_labels[i].set_position((xend + 0.08, yend))
                val_labels[i].set_text(smart_fmt(yend))
            else:
                dots[i].set_data([], [])
                val_labels[i].set_text("")

        # Period label: interpolate between index labels
        period_idx = int(np.clip(round(x_interp[f]), 0, n_periods - 1))
        period_txt.set_text(idx_labels[period_idx])

        return main_lines + glow_lines + dots + val_labels + [period_txt]

    # ── Write video ───────────────────────────────────────────────────────────
    ani = mpl_animation.FuncAnimation(
        fig, update,
        frames=total_frames,
        interval=1000 / FPS,
        blit=True,
    )

    writer = mpl_animation.FFMpegWriter(
        fps=FPS,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18"],
    )
    ani.save(raw_path, writer=writer, dpi=DPI, savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    return raw_path


# ── Music ──────────────────────────────────────────────────────────────────────

def download_music(tmp_dir: str) -> str | None:
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


# ── Post-production (MoviePy 2.x) ─────────────────────────────────────────────

def post_produce(raw_path: str, final_path: str, tmp_dir: str) -> str:
    """Add background music; export clean 1080×1920 MP4."""
    from moviepy import (
        VideoFileClip, AudioFileClip, concatenate_audioclips,
    )

    video = VideoFileClip(raw_path)
    duration = video.duration

    music_path = download_music(tmp_dir)
    if music_path:
        try:
            audio = AudioFileClip(music_path)
            if audio.duration < duration:
                loops = int(np.ceil(duration / audio.duration))
                audio = concatenate_audioclips([audio] * loops)
            audio = audio.subclipped(0, duration).with_volume_scaled(0.32)
            video  = video.with_audio(audio)
        except Exception:
            pass

    video.write_videofile(
        final_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=os.path.join(tmp_dir, "tmp_audio.m4a"),
        remove_temp=True,
        logger=None,
        preset="fast",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "18"],
    )
    video.close()
    return final_path


# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Topic-to-Reel", page_icon="🎬", layout="centered")

st.title("Topic-to-Reel")
st.caption("Enter a topic — get a 1080×1920 animated line chart Reel.")

topic = st.text_area(
    "Topic",
    placeholder=(
        "e.g.  Global male vs female population 2000–2026\n"
        "      US vs China GDP over time\n"
        "      iPhone vs Android market share 2010–2024"
    ),
    height=110,
)

generate = st.button("Generate Reel", type="primary", use_container_width=True)

if generate:
    if not topic.strip():
        st.warning("Please enter a topic first.")
        st.stop()

    progress = st.progress(0, text="Starting…")
    status   = st.empty()

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_mp4   = os.path.join(tmp_dir, "race.mp4")
            final_mp4 = os.path.join(tmp_dir, "reel.mp4")

            # Stage 1 — Research Data
            progress.progress(8,  text="Researching data…")
            status.info("🔍  Researching and structuring data with AI…")
            df, chart_title = extract_data_from_llm(topic)

            # Stage 2 — Generate Animation
            progress.progress(30, text="Data ready — rendering animation…")
            status.info(f"📈  Rendering line race for: **{chart_title}**")
            create_line_race_video(df, chart_title, raw_mp4)

            # Stage 3 — Post-Production
            progress.progress(75, text="Animation done — adding music…")
            status.info("🎬  Adding background music…")
            post_produce(raw_mp4, final_mp4, tmp_dir)

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
