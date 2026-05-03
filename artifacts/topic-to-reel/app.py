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
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

# ── OpenAI client ──────────────────────────────────────────────────────────────
client = OpenAI(
    base_url=os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL"),
    api_key=os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY"),
)

# ── Constants ──────────────────────────────────────────────────────────────────
FONT_BOLD = "/run/current-system/sw/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG  = "/run/current-system/sw/share/fonts/truetype/dejavu/DejaVuSans.ttf"

LINE_COLORS = [
    "#4FC3F7", "#FF7043", "#66BB6A", "#FFA726",
    "#CE93D8", "#26C6DA", "#F06292", "#A5D6A7",
]
MUSIC_URLS = [
    "https://cdn.pixabay.com/download/audio/2021/09/23/audio_57bc8dcb4e.mp3",
    "https://cdn.pixabay.com/download/audio/2022/10/25/audio_d0bba6d4e2.mp3",
    "https://cdn.pixabay.com/download/audio/2022/03/15/audio_c8f4f4a5db.mp3",
]
REDDIT_HEADERS = {"User-Agent": "TopicToReel/1.0"}
FALLBACK_TOPICS = [
    "Global male vs female population 2000–2026",
    "US vs China GDP over time 1980–2024",
    "iPhone vs Android global market share 2010–2024",
    "Top 5 most spoken languages by native speakers over time",
    "Global renewable vs fossil fuel energy 2000–2023",
    "Top social media platforms by monthly active users 2012–2024",
    "CO₂ emissions by continent 1990–2023",
    "Electric vehicle sales by country 2015–2024",
]

BG  = "#000000"
FPS = 30
STEPS_PER_PERIOD = 30
ICON_SIZE = 72   # pixels for circular icon images


# ── Icon helpers ───────────────────────────────────────────────────────────────

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in (FONT_BOLD, FONT_REG):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _make_circle_icon(pil_img: Image.Image, size: int = ICON_SIZE) -> np.ndarray:
    """Resize image and crop to circle → RGBA numpy array."""
    img  = pil_img.convert("RGBA").resize((size, size), Image.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
    img.putalpha(mask)
    return np.array(img)


def _make_initials_icon(color_hex: str, label: str, size: int = ICON_SIZE) -> np.ndarray:
    """Colored circle with first letter — fallback when no image is available."""
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
    # Outer ring
    draw.ellipse((0, 0, size - 1, size - 1), fill=(r, g, b, 230))
    # Letter
    letter   = label[0].upper()
    font     = _load_font(max(16, size // 2))
    bbox     = draw.textbbox((0, 0), letter, font=font)
    tw, th   = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size - tw) // 2, (size - th) // 2), letter, fill=(255, 255, 255, 255), font=font)
    return np.array(img)


def identify_country_codes(cols: list[str]) -> dict[str, str | None]:
    """Ask LLM to map column names → ISO-2 country codes (or None if not a country)."""
    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{
            "role": "user",
            "content": (
                f"Given these chart category names: {cols}\n"
                "For each name that is clearly a country, return its ISO 3166-1 alpha-2 code "
                "(lowercase). For everything else return null.\n"
                "Return ONLY a JSON object like: "
                '{\"USA\": \"us\", \"China\": \"cn\", \"Male\": null}\n'
                "No explanation, no markdown."
            ),
        }],
        max_completion_tokens=120,
    )
    import json
    try:
        raw = resp.choices[0].message.content.strip()
        raw = raw.strip("```json").strip("```").strip()
        return json.loads(raw)
    except Exception:
        return {c: None for c in cols}


def get_icons(cols: list[str], colors: list[str]) -> list[np.ndarray]:
    """Return a circular icon (RGBA numpy array) for each column."""
    country_map = identify_country_codes(cols)
    icons: list[np.ndarray] = []
    for i, col in enumerate(cols):
        code = country_map.get(col)
        icon = None
        if code:
            try:
                url = f"https://flagcdn.com/48x36/{code}.png"
                r   = requests.get(url, timeout=6)
                if r.status_code == 200:
                    pil = Image.open(io.BytesIO(r.content))
                    icon = _make_circle_icon(pil)
            except Exception:
                pass
        if icon is None:
            icon = _make_initials_icon(colors[i], col)
        icons.append(icon)
    return icons


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
    df      = df.apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0)

    title_resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content":
            f"Write a short chart title (max 7 words, title case) for: {topic}. Return only the title."}],
        max_completion_tokens=30,
    )
    chart_title = title_resp.choices[0].message.content.strip().strip('"\'')
    return df, chart_title


# ── Caption generation ─────────────────────────────────────────────────────────

def generate_caption(topic: str, chart_title: str) -> tuple[str, list[str]]:
    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{
            "role": "user",
            "content": (
                f"Write an engaging Instagram caption (3-5 sentences) for a data visualization reel about: '{topic}' "
                f"titled '{chart_title}'. Be conversational, insightful, end with a question to boost engagement. "
                "Then on a new line write exactly: HASHTAGS: #tag1 #tag2 #tag3 #tag4 #tag5\n"
                "Return ONLY the caption text and the HASHTAGS line, nothing else."
            ),
        }],
        max_completion_tokens=300,
    )
    text = resp.choices[0].message.content.strip()
    caption, hashtags_str = text, ""
    if "HASHTAGS:" in text:
        parts       = text.split("HASHTAGS:")
        caption     = parts[0].strip()
        hashtags_str = parts[1].strip()
    hashtags = [h.strip().lstrip("#") for h in hashtags_str.split() if h.startswith("#")][:5]
    return caption, hashtags


# ── Smart number formatter ─────────────────────────────────────────────────────

def smart_fmt(val: float) -> str:
    if abs(val) >= 1_000_000_000:
        return f"{val / 1_000_000_000:.2f}B"
    if abs(val) >= 1_000_000:
        return f"{val / 1_000_000:.2f}M"
    if abs(val) >= 10_000:
        return f"{val:,.0f}"
    if abs(val) >= 100:
        return f"{val:.0f}"
    return f"{val:.2f}"


# ── Animation ──────────────────────────────────────────────────────────────────

def _build_figure(df: pd.DataFrame, chart_title: str, icon_arrays: list[np.ndarray],
                  n_periods: int, cols: list[str], colors: list[str],
                  y_interp: dict, idx_labels: list[str]):
    """Build the matplotlib figure and all artists. Returns (fig, ax, artists_dict)."""
    FIG_W, FIG_H = 6.75, 12.0
    n_lines = len(cols)

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)

    gs = fig.add_gridspec(
        2, 1, height_ratios=[0.11, 0.89],
        left=0.13, right=0.72, top=0.91, bottom=0.09, hspace=0.02,
    )
    title_ax = fig.add_subplot(gs[0])
    ax       = fig.add_subplot(gs[1])

    for a in (title_ax, ax):
        a.set_facecolor(BG)
        a.patch.set_facecolor(BG)

    # ── Title strip ───────────────────────────────────────────────────────────
    title_ax.axis("off")
    title_ax.text(0.5, 0.72, chart_title,
                  transform=title_ax.transAxes,
                  color="white", fontsize=16, fontweight="bold",
                  ha="center", va="center", fontfamily="DejaVu Sans")
    title_ax.text(0.5, 0.22, "↓  Read caption to know more  ↓",
                  transform=title_ax.transAxes,
                  color="#666666", fontsize=8.5, ha="center", va="center",
                  fontstyle="italic", fontfamily="DejaVu Sans")

    # ── Chart styling ─────────────────────────────────────────────────────────
    ax.tick_params(colors="#555555", labelsize=0)  # hide default tick labels
    ax.set_xticks([])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#222222")
        ax.spines[spine].set_linewidth(0.8)
    ax.yaxis.grid(True, color="#141414", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # ── Axes limits — compressing x: normalized [0,1], icons at x=1 ──────────
    all_y  = np.concatenate([y_interp[c] for c in cols])
    y_min, y_max = all_y.min(), all_y.max()
    y_pad  = (y_max - y_min) * 0.10
    # x: 0=start, 1=current (normalized). Extra right space for labels
    ax.set_xlim(-0.03, 1.38)
    ax.set_ylim(y_min - y_pad, y_max + y_pad * 2.5)

    # Y-axis formatter
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: smart_fmt(v)))
    ax.tick_params(axis="y", labelcolor="#555555", labelsize=8.5)

    # ── Custom x-axis tick labels (will update each frame) ────────────────────
    TICK_NORM = [0.0, 0.25, 0.5, 0.75, 1.0]
    tick_texts = []
    for tx in TICK_NORM:
        t = ax.text(tx, 0, "", color="#555555", fontsize=8,
                    ha="center", va="top",
                    transform=ax.get_xaxis_transform(),
                    fontfamily="DejaVu Sans")
        tick_texts.append(t)
    # Small tick marks
    for tx in TICK_NORM:
        ax.axvline(tx, ymin=0, ymax=0.01, color="#333333", linewidth=0.8)

    # ── Per-series artists ────────────────────────────────────────────────────
    main_lines, glow_lines, anno_boxes, val_labels = [], [], [], []
    for i, col in enumerate(cols):
        c = colors[i]
        glow, = ax.plot([], [], color=c, linewidth=10, alpha=0.10,
                        solid_capstyle="round", zorder=2)
        main, = ax.plot([], [], color=c, linewidth=2.2,
                        solid_capstyle="round", zorder=3)

        # Icon (AnnotationBbox)
        zoom = 0.42 if icon_arrays[i].shape[-1] == 4 else 0.40
        imagebox = OffsetImage(icon_arrays[i], zoom=zoom)
        ab = AnnotationBbox(imagebox, (0, 0), frameon=False, zorder=10,
                            clip_on=False, pad=0)
        ab.set_visible(False)
        ax.add_artist(ab)

        # Value label  "Name: value"
        lbl = ax.text(0, 0, "", color=c, fontsize=9, fontweight="bold",
                      va="center", ha="left", zorder=11, clip_on=False,
                      fontfamily="DejaVu Sans")

        glow_lines.append(glow)
        main_lines.append(main)
        anno_boxes.append(ab)
        val_labels.append(lbl)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        plt.Line2D([0], [0], color=colors[i], linewidth=2.2, label=cols[i])
        for i in range(n_lines)
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              frameon=True, framealpha=0.18,
              facecolor="#0d0d0d", edgecolor="#222222",
              labelcolor="white", fontsize=9,
              handlelength=1.5, borderpad=0.6, labelspacing=0.4)

    # ── Period counter ────────────────────────────────────────────────────────
    period_txt = ax.text(0.70, 0.06, "",
                         transform=ax.transAxes,
                         color="white", fontsize=34, fontweight="bold",
                         ha="right", va="bottom", alpha=0.75,
                         fontfamily="DejaVu Sans")

    # ── Watermark ─────────────────────────────────────────────────────────────
    fig.text(0.5, 0.022, "randomdatavstime",
             ha="center", va="bottom", fontsize=7.5,
             color="#2e2e2e", fontfamily="DejaVu Sans")

    artists = dict(
        main_lines=main_lines, glow_lines=glow_lines,
        anno_boxes=anno_boxes, val_labels=val_labels,
        tick_texts=tick_texts, period_txt=period_txt,
    )
    return fig, ax, artists, TICK_NORM


def create_line_race_video(df: pd.DataFrame, chart_title: str,
                           icon_arrays: list[np.ndarray], raw_path: str) -> str:

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

    DPI = 160
    fig, ax, artists, TICK_NORM = _build_figure(
        df, chart_title, icon_arrays,
        n_periods, cols, colors, y_interp, idx_labels,
    )
    main_lines = artists["main_lines"]
    glow_lines = artists["glow_lines"]
    anno_boxes = artists["anno_boxes"]
    val_labels = artists["val_labels"]
    tick_texts = artists["tick_texts"]
    period_txt = artists["period_txt"]

    def update(frame: int):
        f          = min(frame, total_frames - 1)
        current_x  = x_interp[f]          # raw index position (0 → n_periods-1)
        p_idx      = int(np.clip(round(current_x), 0, n_periods - 1))

        # Normalize x coords so current position = 1.0 (compressing x-axis)
        if current_x > 0:
            x_norm = x_interp[:f + 1] / current_x
            icon_x = 1.0
        else:
            x_norm = np.array([0.0])
            icon_x = 0.0

        for i, col in enumerate(cols):
            y_now = y_interp[col][:f + 1]
            main_lines[i].set_data(x_norm, y_now)
            glow_lines[i].set_data(x_norm, y_now)

            if len(y_now) > 0:
                yend = y_now[-1]
                anno_boxes[i].xy = (icon_x, yend)
                anno_boxes[i].set_visible(True)
                val_labels[i].set_position((icon_x + 0.055, yend))
                val_labels[i].set_text(f"{col}: {smart_fmt(yend)}")
            else:
                anno_boxes[i].set_visible(False)
                val_labels[i].set_text("")

        # Dynamic x-tick labels
        for j, txt_obj in enumerate(tick_texts):
            mapped = min(int(TICK_NORM[j] * p_idx), n_periods - 1)
            txt_obj.set_text(idx_labels[mapped])

        period_txt.set_text(idx_labels[p_idx])

    ani = mpl_animation.FuncAnimation(
        fig, update, frames=total_frames,
        interval=1000 / FPS, blit=False,
    )
    writer = mpl_animation.FFMpegWriter(
        fps=FPS, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18"],
    )
    ani.save(raw_path, writer=writer, dpi=DPI, savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    return raw_path


# ── Thumbnail ──────────────────────────────────────────────────────────────────

def create_thumbnail(df: pd.DataFrame, chart_title: str,
                     icon_arrays: list[np.ndarray]) -> bytes:
    """Generate a minimalistic 9:16 thumbnail of the final chart state."""
    n_lines = min(len(df.columns), len(LINE_COLORS))
    df      = df.iloc[:, :n_lines]
    cols    = list(df.columns)
    colors  = LINE_COLORS[:n_lines]
    n_periods = len(df)

    x_raw = np.arange(n_periods, dtype=float)
    # Final state: all data visible, x normalized to [0,1]
    x_norm = x_raw / (n_periods - 1) if n_periods > 1 else x_raw
    y_data = {c: df[c].values for c in cols}
    idx_labels = [str(v) for v in df.index]

    fig = plt.figure(figsize=(6.75, 12.0), facecolor=BG, dpi=120)
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.22, 0.78],
                            left=0.13, right=0.72, top=0.91, bottom=0.09, hspace=0.03)
    title_ax = fig.add_subplot(gs[0])
    ax       = fig.add_subplot(gs[1])
    for a in (title_ax, ax):
        a.set_facecolor(BG); a.patch.set_facecolor(BG)

    # Title block
    title_ax.axis("off")
    title_ax.text(0.5, 0.70, chart_title,
                  transform=title_ax.transAxes,
                  color="white", fontsize=18, fontweight="bold",
                  ha="center", va="center", fontfamily="DejaVu Sans")
    # Hook text
    hook_lines = ["Who's leading?", "The answer might", "surprise you 👆"]
    title_ax.text(0.5, 0.24, "  ".join(hook_lines),
                  transform=title_ax.transAxes,
                  color="#AAAAAA", fontsize=9, ha="center", va="center",
                  fontfamily="DejaVu Sans", fontstyle="italic")

    # Chart
    ax.set_facecolor(BG)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#222222")
    ax.yaxis.grid(True, color="#141414", linewidth=0.8); ax.set_axisbelow(True)
    ax.set_xticks([]); ax.set_xlim(-0.03, 1.38)

    all_y = np.concatenate([y_data[c] for c in cols])
    y_min, y_max = all_y.min(), all_y.max()
    y_pad = (y_max - y_min) * 0.10
    ax.set_ylim(y_min - y_pad, y_max + y_pad * 2.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: smart_fmt(v)))
    ax.tick_params(axis="y", labelcolor="#555555", labelsize=8.5)

    for i, col in enumerate(cols):
        c = colors[i]
        ax.plot(x_norm, y_data[col], color=c, linewidth=2.5, solid_capstyle="round", zorder=3)
        ax.plot(x_norm, y_data[col], color=c, linewidth=10, alpha=0.10,
                solid_capstyle="round", zorder=2)

        yend = y_data[col][-1]
        # Icon at x=1.0
        zoom = 0.38
        imagebox = OffsetImage(icon_arrays[i], zoom=zoom)
        ab = AnnotationBbox(imagebox, (1.0, yend), frameon=False, zorder=10,
                            clip_on=False, pad=0)
        ax.add_artist(ab)
        # Label
        ax.text(1.05, yend, f"{col}: {smart_fmt(yend)}",
                color=c, fontsize=8.5, fontweight="bold",
                va="center", ha="left", clip_on=False, zorder=11,
                fontfamily="DejaVu Sans")

    # X-tick labels (fixed, full range)
    for tx in [0.0, 0.25, 0.5, 0.75, 1.0]:
        mapped = min(int(tx * (n_periods - 1)), n_periods - 1)
        ax.text(tx, 0, idx_labels[mapped], color="#555555", fontsize=8,
                ha="center", va="top", transform=ax.get_xaxis_transform(),
                fontfamily="DejaVu Sans")

    # Watermark
    fig.text(0.5, 0.022, "randomdatavstime", ha="center", va="bottom",
             fontsize=7.5, color="#2e2e2e", fontfamily="DejaVu Sans")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG, bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


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


# ── Post-production ────────────────────────────────────────────────────────────

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
        final_path, fps=FPS, codec="libx264", audio_codec="aac",
        temp_audiofile=os.path.join(tmp_dir, "tmp_audio.m4a"),
        remove_temp=True, logger=None, preset="fast",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "18"],
    )
    video.close()
    return final_path


# ── Trending topics ────────────────────────────────────────────────────────────

def _fetch_reddit_titles(sub: str, limit: int = 6) -> list[str]:
    try:
        r = requests.get(f"https://www.reddit.com/r/{sub}/hot.json?limit={limit}",
                         headers=REDDIT_HEADERS, timeout=8)
        if r.status_code == 200:
            return [p["data"]["title"] for p in r.json()["data"]["children"]
                    if not p["data"].get("stickied")]
    except Exception:
        pass
    return []


@st.cache_data(ttl=3600, show_spinner=False)
def get_trending_topics() -> list[str]:
    raw: list[str] = []
    for sub in ["dataisbeautiful", "worldnews", "science", "technology", "economics"]:
        raw.extend(_fetch_reddit_titles(sub, 5))
        if len(raw) >= 20:
            break
    if not raw:
        return FALLBACK_TOPICS
    block = "\n".join(f"- {t}" for t in raw[:25])
    try:
        resp = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": (
                    "Convert trending headlines into 8 animated line chart prompts "
                    "(comparisons over time). One per line, no bullets, no numbers."
                )},
                {"role": "user", "content": f"Trending:\n{block}\n\nGenerate 8 chart prompts."},
            ],
            max_completion_tokens=300,
        )
        lines = [l.strip() for l in resp.choices[0].message.content.strip().splitlines() if l.strip()]
        if len(lines) >= 4:
            return lines[:8]
    except Exception:
        pass
    return FALLBACK_TOPICS


# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Topic-to-Reel", page_icon="🎬", layout="centered")
st.title("Topic-to-Reel")
st.caption("Enter a topic — get a 1080×1920 animated line chart Reel.")

# ── Trending suggestions ───────────────────────────────────────────────────────
with st.spinner("Fetching today's trending topics…"):
    suggestions = get_trending_topics()

st.markdown("**Trending today — tap to use:**")

# Session-state backed topic value (fixes the "please enter topic" bug)
if "topic_value" not in st.session_state:
    st.session_state["topic_value"] = ""

btn_cols = st.columns(2)
for idx, sug in enumerate(suggestions):
    if btn_cols[idx % 2].button(sug, key=f"sug_{idx}", use_container_width=True):
        st.session_state["topic_value"] = sug
        st.rerun()

st.divider()

# Text area bound to session state key — value persists on rerun
topic = st.text_area(
    "Or type your own topic",
    key="topic_value",
    placeholder=(
        "e.g.  Global male vs female population 2000–2026\n"
        "      US vs China GDP over time\n"
        "      iPhone vs Android market share 2010–2024"
    ),
    height=90,
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

            # Stage 1 — Data
            progress.progress(5, text="Researching data…")
            status.info("🔍  Researching and structuring data with AI…")
            df, chart_title = extract_data_from_llm(topic)

            # Stage 2 — Icons
            progress.progress(18, text="Fetching icons…")
            status.info("🏳  Fetching icons for each category…")
            n_lines = min(len(df.columns), len(LINE_COLORS))
            cols_used = list(df.columns[:n_lines])
            colors_used = LINE_COLORS[:n_lines]
            icon_arrays = get_icons(cols_used, colors_used)

            # Stage 3 — Animation
            progress.progress(28, text="Rendering animation…")
            status.info(f"📈  Rendering compressing line race for: **{chart_title}**")
            create_line_race_video(df, chart_title, icon_arrays, raw_mp4)

            # Stage 4 — Music
            progress.progress(78, text="Adding music…")
            status.info("🎬  Adding background music…")
            post_produce(raw_mp4, final_mp4, tmp_dir)

            # Stage 5 — Caption
            progress.progress(90, text="Generating caption…")
            status.info("✍️  Writing caption and hashtags…")
            caption, hashtags = generate_caption(topic, chart_title)

            # Stage 6 — Thumbnail
            progress.progress(96, text="Creating thumbnail…")
            status.info("🖼  Creating thumbnail…")
            thumb_bytes = create_thumbnail(df, chart_title, icon_arrays)

            progress.progress(100, text="Done!")
            status.success("✅  Your Reel is ready!")

            with open(final_mp4, "rb") as f:
                video_bytes = f.read()

        # ── Output ────────────────────────────────────────────────────────────
        st.video(video_bytes)
        st.download_button(
            label="⬇️  Download MP4 (1080×1920)",
            data=video_bytes,
            file_name=f"reel_{int(time.time())}.mp4",
            mime="video/mp4",
            use_container_width=True,
        )

        # Thumbnail
        st.subheader("Thumbnail")
        st.image(thumb_bytes, use_container_width=True)
        st.download_button(
            label="⬇️  Download Thumbnail (PNG)",
            data=thumb_bytes,
            file_name=f"thumb_{int(time.time())}.png",
            mime="image/png",
            use_container_width=True,
        )

        # Caption + hashtags
        st.subheader("Caption")
        hashtag_line = " ".join(f"#{h}" for h in hashtags)
        full_caption = f"{caption}\n\n{hashtag_line}"
        st.code(full_caption, language=None)   # st.code has built-in copy button

    except Exception as exc:
        progress.empty()
        status.empty()
        st.error(f"Something went wrong: {exc}")
        with st.expander("Error details"):
            st.exception(exc)
