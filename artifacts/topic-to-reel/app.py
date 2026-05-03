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
from openai import OpenAI

# ── OpenAI client ──────────────────────────────────────────────────────────────
client = OpenAI(
    base_url=os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL"),
    api_key=os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY"),
)

# ── Constants ──────────────────────────────────────────────────────────────────
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

REDDIT_HEADERS = {"User-Agent": "TopicToReel/1.0 (data viz reel generator)"}
BG  = "#000000"
FPS = 30
STEPS_PER_PERIOD = 30


# ── Helpers ────────────────────────────────────────────────────────────────────

def smart_fmt(val: float) -> str:
    if abs(val) >= 1_000_000_000:
        return f"{val/1_000_000_000:.2f}B"
    if abs(val) >= 1_000_000:
        return f"{val/1_000_000:.2f}M"
    if abs(val) >= 10_000:
        return f"{val:,.0f}"
    if abs(val) >= 100:
        return f"{val:.0f}"
    return f"{val:.2f}"


# ── Trending topics ────────────────────────────────────────────────────────────

FALLBACK_TOPICS = [
    "Global male vs female population 2000–2026",
    "US vs China GDP over time 1980–2024",
    "iPhone vs Android global market share 2010–2024",
    "World's top 5 most spoken languages by speakers over time",
    "Global renewable vs fossil fuel energy production 2000–2023",
    "Top social media platforms by monthly active users 2010–2024",
    "CO₂ emissions by continent 1990–2023",
    "Electric vehicle sales by country 2015–2024",
]


def _fetch_reddit_titles(subreddit: str, limit: int = 8) -> list[str]:
    try:
        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
        r = requests.get(url, headers=REDDIT_HEADERS, timeout=8)
        if r.status_code == 200:
            posts = r.json()["data"]["children"]
            return [p["data"]["title"] for p in posts if not p["data"].get("stickied")]
    except Exception:
        pass
    return []


@st.cache_data(ttl=3600, show_spinner=False)
def get_trending_topics() -> list[str]:
    """Scrape Reddit for trending topics and convert to viz-ready prompts."""
    raw_titles: list[str] = []
    for sub in ["dataisbeautiful", "worldnews", "science", "technology", "economics"]:
        raw_titles.extend(_fetch_reddit_titles(sub, limit=5))
        if len(raw_titles) >= 20:
            break

    if not raw_titles:
        return FALLBACK_TOPICS

    titles_block = "\n".join(f"- {t}" for t in raw_titles[:25])
    try:
        resp = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": (
                    "You convert trending news headlines into concise data visualization prompts "
                    "suitable for animated line chart race videos. Each prompt should describe "
                    "a comparison over time (e.g. 'US vs China GDP 1990–2024'). "
                    "Return exactly 8 prompts, one per line, no numbering, no bullets."
                )},
                {"role": "user", "content": (
                    f"Today's trending topics:\n{titles_block}\n\n"
                    "Generate 8 animated chart prompts inspired by these trends."
                )},
            ],
            max_completion_tokens=300,
        )
        lines = [l.strip() for l in resp.choices[0].message.content.strip().splitlines() if l.strip()]
        if len(lines) >= 4:
            return lines[:8]
    except Exception:
        pass

    return FALLBACK_TOPICS


# ── Data extraction ────────────────────────────────────────────────────────────

def extract_data_from_llm(topic: str) -> tuple[pd.DataFrame, str]:
    prompt = f"""You are a data research assistant producing data for an animated line chart.

Topic: "{topic}"

Rules:
- First column: numeric time index (Year as integer, e.g. 2000, 2001, …)
- 2 to 6 category columns, short names (≤18 chars)
- Numeric values only — no commas inside numbers, no units, no currency symbols
- ONE ROW PER YEAR — do NOT skip years
- Minimum 15 rows, ideally 20–30 rows
- Values must change realistically over time
- No missing values

Return ONLY raw CSV — no markdown, no explanation, no code fences.

Example:
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
    raw     = resp.choices[0].message.content.strip()
    cleaned = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("```"))
    df      = pd.read_csv(io.StringIO(cleaned))
    df      = df.set_index(df.columns[0])
    df      = df.apply(pd.to_numeric, errors="coerce")
    df      = df.ffill().bfill().fillna(0)

    title_resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content":
            f"Write a short chart title (max 7 words, title case) for: {topic}. Return only the title."}],
        max_completion_tokens=30,
    )
    chart_title = title_resp.choices[0].message.content.strip().strip('"\'')
    return df, chart_title


# ── Line race animation ────────────────────────────────────────────────────────

def create_line_race_video(df: pd.DataFrame, chart_title: str, raw_path: str) -> str:
    """Render a growing-line race to 9:16 MP4 (1080×1920)."""

    n_periods = len(df)
    n_lines   = min(len(df.columns), len(LINE_COLORS))
    df        = df.iloc[:, :n_lines]
    cols      = list(df.columns)
    colors    = LINE_COLORS[:n_lines]

    total_frames = (n_periods - 1) * STEPS_PER_PERIOD + 1
    x_raw        = np.arange(n_periods, dtype=float)
    x_interp     = np.linspace(0, n_periods - 1, total_frames)
    y_interp     = {c: np.interp(x_interp, x_raw, df[c].values) for c in cols}
    idx_labels   = [str(v) for v in df.index]

    # ── Figure — 1080×1920 @ 160 DPI ─────────────────────────────────────────
    FIG_W, FIG_H, DPI = 6.75, 12.0, 160
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG, dpi=DPI)

    # Two rows only — title strip + chart.
    # Generous margins on all sides = "zoomed out" feel with clean black border.
    gs = fig.add_gridspec(
        nrows=2, ncols=1,
        height_ratios=[0.08, 0.92],
        left=0.13,    # left border
        right=0.79,   # right border (leaves room for value labels)
        top=0.91,     # top border
        bottom=0.09,  # bottom border
        hspace=0.02,
    )
    title_ax = fig.add_subplot(gs[0])
    ax       = fig.add_subplot(gs[1])

    # Both axes: fully black, no artefacts
    for a in (title_ax, ax):
        a.set_facecolor(BG)
        a.patch.set_facecolor(BG)

    # ── Title strip ───────────────────────────────────────────────────────────
    title_ax.axis("off")
    title_ax.text(
        0.5, 0.5, chart_title,
        transform=title_ax.transAxes,
        color="white", fontsize=17, fontweight="bold",
        ha="center", va="center",
        fontfamily="DejaVu Sans",
    )

    # ── Chart styling ─────────────────────────────────────────────────────────
    ax.tick_params(colors="#666666", labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#2a2a2a")
        ax.spines[spine].set_linewidth(0.8)

    ax.yaxis.grid(True, color="#181818", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)

    # ── Axes limits ───────────────────────────────────────────────────────────
    all_y  = np.concatenate([y_interp[c] for c in cols])
    y_min, y_max = all_y.min(), all_y.max()
    y_pad  = (y_max - y_min) * 0.10
    ax.set_xlim(-0.3, n_periods - 0.7)
    ax.set_ylim(y_min - y_pad, y_max + y_pad * 2.8)

    # X-axis ticks
    tick_step = max(1, n_periods // 6)
    tick_pos  = list(range(0, n_periods, tick_step))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([idx_labels[i] for i in tick_pos], color="#666666", fontsize=9)

    # Y-axis formatter
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: smart_fmt(v)))
    ax.tick_params(axis="y", labelcolor="#666666", labelsize=9)

    # ── Per-series artists ────────────────────────────────────────────────────
    main_lines, glow_lines, dots, val_labels = [], [], [], []
    for i, col in enumerate(cols):
        c     = colors[i]
        glow, = ax.plot([], [], color=c, linewidth=11, alpha=0.12, solid_capstyle="round", zorder=2)
        main, = ax.plot([], [], color=c, linewidth=2.5, solid_capstyle="round", zorder=3)
        dot,  = ax.plot([], [], "o", color=c, markersize=8, zorder=5)
        lbl   = ax.text(0, 0, "", color=c, fontsize=10, fontweight="bold",
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
        framealpha=0.18,
        facecolor="#0d0d0d",
        edgecolor="#2a2a2a",
        labelcolor="white",
        fontsize=10,
        handlelength=1.6,
        borderpad=0.7,
        labelspacing=0.5,
    )

    # ── Period counter ────────────────────────────────────────────────────────
    period_txt = ax.text(
        0.97, 0.05, "",
        transform=ax.transAxes,
        color="white", fontsize=38, fontweight="bold",
        ha="right", va="bottom", alpha=0.80,
        fontfamily="DejaVu Sans",
    )

    # ── Watermark ─────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.025, "randomdatavstime",
        ha="center", va="bottom", fontsize=8,
        color="#333333", fontfamily="DejaVu Sans",
    )

    # ── Animation ─────────────────────────────────────────────────────────────
    def update(frame: int):
        f     = min(frame, total_frames - 1)
        x_now = x_interp[:f + 1]

        for i, col in enumerate(cols):
            y_now = y_interp[col][:f + 1]
            main_lines[i].set_data(x_now, y_now)
            glow_lines[i].set_data(x_now, y_now)

            if len(x_now) > 0:
                xend, yend = x_now[-1], y_now[-1]
                dots[i].set_data([xend], [yend])
                val_labels[i].set_position((xend + 0.12, yend))
                val_labels[i].set_text(smart_fmt(yend))
            else:
                dots[i].set_data([], [])
                val_labels[i].set_text("")

        period_idx = int(np.clip(round(x_interp[f]), 0, n_periods - 1))
        period_txt.set_text(idx_labels[period_idx])

        return main_lines + glow_lines + dots + val_labels + [period_txt]

    ani = mpl_animation.FuncAnimation(
        fig, update,
        frames=total_frames,
        interval=1000 / FPS,
        blit=True,
    )
    writer = mpl_animation.FFMpegWriter(
        fps=FPS, codec="libx264",
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
    from moviepy import VideoFileClip, AudioFileClip, concatenate_audioclips

    video    = VideoFileClip(raw_path)
    duration = video.duration

    music_path = download_music(tmp_dir)
    if music_path:
        try:
            audio = AudioFileClip(music_path)
            if audio.duration < duration:
                loops = int(np.ceil(duration / audio.duration))
                audio = concatenate_audioclips([audio] * loops)
            audio = audio.subclipped(0, duration).with_volume_scaled(0.32)
            video = video.with_audio(audio)
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

# ── Trending suggestions ───────────────────────────────────────────────────────
with st.spinner("Fetching today's trending topics…"):
    suggestions = get_trending_topics()

st.markdown("**Trending today — tap to use:**")

# Render suggestion buttons in a 2-column grid
cols_ui = st.columns(2)
for idx, suggestion in enumerate(suggestions):
    if cols_ui[idx % 2].button(suggestion, key=f"sug_{idx}", use_container_width=True):
        st.session_state["topic_prefill"] = suggestion
        st.rerun()

st.divider()

# ── Topic input ───────────────────────────────────────────────────────────────
prefill = st.session_state.pop("topic_prefill", "") if "topic_prefill" in st.session_state else ""

topic = st.text_area(
    "Or type your own topic",
    value=prefill,
    placeholder=(
        "e.g.  Global male vs female population 2000–2026\n"
        "      US vs China GDP over time\n"
        "      iPhone vs Android market share 2010–2024"
    ),
    height=100,
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

            progress.progress(8,  text="Researching data…")
            status.info("🔍  Researching and structuring data with AI…")
            df, chart_title = extract_data_from_llm(topic)

            progress.progress(30, text="Data ready — rendering animation…")
            status.info(f"📈  Rendering line race for: **{chart_title}**")
            create_line_race_video(df, chart_title, raw_mp4)

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
