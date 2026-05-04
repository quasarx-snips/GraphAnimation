import shutil
import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import io
import json
import time
import requests
import matplotlib
matplotlib.use("Agg")
_ffmpeg = shutil.which("ffmpeg")
if _ffmpeg:
    matplotlib.rcParams["animation.ffmpeg_path"] = _ffmpeg
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation
import matplotlib.ticker as mticker
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

# ── Client ─────────────────────────────────────────────────────────────────────
client = OpenAI(
    base_url=os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL"),
    api_key=os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY"),
)

# ── Constants ──────────────────────────────────────────────────────────────────
FONT_BOLD = "/run/current-system/sw/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG  = "/run/current-system/sw/share/fonts/truetype/dejavu/DejaVuSans.ttf"
LINE_COLORS = [
    "#4FC3F7", "#FF7043", "#69F0AE", "#FFD740",
    "#E040FB", "#40C4FF", "#FF6D00", "#B9F6CA",
]
MUSIC_URLS = [
    "https://cdn.pixabay.com/download/audio/2021/09/23/audio_57bc8dcb4e.mp3",
    "https://cdn.pixabay.com/download/audio/2022/10/25/audio_d0bba6d4e2.mp3",
]
REDDIT_UA = {"User-Agent": "TopicToReel/1.0"}
BRAND     = "worldstats.visualised"
BG        = "#000000"
FPS       = 30
ICON_PX   = 54       # target icon size in video pixels
# Gridspec constants (used for icon sizing)
AX_L, AX_R, AX_T, AX_B = 0.12, 0.66, 0.91, 0.09

FALLBACK_TOPICS = [
    "US vs China GDP 1980–2024",
    "Global male vs female population 2000–2026",
    "iPhone vs Android market share 2010–2024",
    "Top social media platforms by users 2010–2024",
    "Global CO₂ emissions by continent 1990–2023",
    "EV sales by country 2015–2024",
    "Netflix vs YouTube vs TikTok subscribers 2015–2024",
    "Global renewable vs fossil fuel energy 2000–2023",
]

# ── Font ───────────────────────────────────────────────────────────────────────
def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in (FONT_BOLD, FONT_REG):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()

# ── Icon helpers ───────────────────────────────────────────────────────────────
def _to_circle(pil_img: Image.Image, size: int = 64) -> np.ndarray:
    img  = pil_img.convert("RGBA").resize((size, size), Image.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
    img.putalpha(mask)
    return np.array(img)

def _initials(color_hex: str, label: str, size: int = 64) -> np.ndarray:
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
    draw.ellipse((0, 0, size - 1, size - 1), fill=(r, g, b, 215))
    letter = label[0].upper()
    font   = _font(max(14, size // 2))
    bbox   = draw.textbbox((0, 0), letter, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size - tw) // 2 - bbox[0], (size - th) // 2 - bbox[1]),
              letter, fill=(255, 255, 255, 255), font=font)
    return np.array(img)

def _flag(code: str, size: int = 64) -> np.ndarray | None:
    try:
        r = requests.get(f"https://flagcdn.com/48x36/{code.lower()}.png", timeout=6)
        if r.status_code == 200:
            return _to_circle(Image.open(io.BytesIO(r.content)), size)
    except Exception:
        pass
    return None

def _identify_countries(cols: list[str]) -> dict[str, str | None]:
    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": (
            f"For each category name {cols}, if it is a country return its ISO 3166-1 alpha-2 "
            "code (lowercase), else return null. ONLY JSON, no markdown. "
            'Example: {"India": "in", "USA": "us", "Population": null}'
        )}],
        max_completion_tokens=120,
    )
    try:
        raw = resp.choices[0].message.content.strip().strip("```json").strip("```")
        return json.loads(raw)
    except Exception:
        return {c: None for c in cols}

def get_icons(cols: list[str], colors: list[str], size: int = 64) -> list[np.ndarray]:
    cmap = _identify_countries(cols)
    out  = []
    for i, col in enumerate(cols):
        code = cmap.get(col)
        icon = _flag(code, size) if code else None
        out.append(icon if icon is not None else _initials(colors[i], col, size))
    return out

# ── Data helpers ───────────────────────────────────────────────────────────────
def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index(df.columns[0]) if not df.index.name else df
    df = df.apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0)
    return df

def extract_data_from_llm(topic: str) -> tuple[pd.DataFrame, str]:
    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": "Return ONLY clean CSV. No markdown. No explanations."},
            {"role": "user", "content": (
                f'Topic: "{topic}"\n'
                "Rules: first col = integer year (no gaps), 2–6 category cols (≤18 chars each), "
                "raw numeric values only, 15–30 rows, realistic trends, no missing values. "
                "Return ONLY the CSV."
            )},
        ],
        max_completion_tokens=4000,
    )
    raw     = resp.choices[0].message.content.strip()
    cleaned = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("```"))
    df      = pd.read_csv(io.StringIO(cleaned))
    df      = _clean_df(df)

    t = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user",
                   "content": f"Short chart title (≤7 words, title case) for: {topic}. Return only the title."}],
        max_completion_tokens=30,
    )
    return df, t.choices[0].message.content.strip().strip("\"'")

def parse_uploaded_file(file) -> tuple[pd.DataFrame, str]:
    name = file.name.lower()
    if name.endswith(".csv") or name.endswith(".txt"):
        df = pd.read_csv(file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(file)
    else:
        raise ValueError(f"Unsupported file type: {file.name}")
    df = _clean_df(df)
    title = file.name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
    return df, title

def parse_pasted_data(raw_text: str) -> tuple[pd.DataFrame, str]:
    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": "Return ONLY clean CSV. No markdown. No explanations."},
            {"role": "user", "content": (
                "Convert this data into a clean CSV:\n"
                "- First column: time index (Year/Month/etc.) as integers or strings\n"
                "- Remaining columns: numeric series\n"
                "- Raw numeric values only, no units in cells\n\n"
                f"Data:\n{raw_text}\n\nReturn ONLY the CSV."
            )},
        ],
        max_completion_tokens=2000,
    )
    raw     = resp.choices[0].message.content.strip()
    cleaned = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("```"))
    df      = pd.read_csv(io.StringIO(cleaned))
    df      = _clean_df(df)

    t = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user",
                   "content": f"Short chart title (≤7 words) for columns {list(df.columns)}. Return only the title."}],
        max_completion_tokens=30,
    )
    return df, t.choices[0].message.content.strip().strip("\"'")

def detect_units(topic: str, df: pd.DataFrame) -> dict:
    sample = {c: {"first": float(df[c].iloc[0]), "last": float(df[c].iloc[-1])}
              for c in list(df.columns)[:3]}
    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": (
            f'Topic: "{topic}", sample values: {json.dumps(sample)}.\n'
            'Return ONLY JSON: {"prefix":"$","suffix":"","description":"USD Billions"}\n'
            "prefix = currency symbol or empty, suffix = % / ppl / etc. or empty, "
            "description = short human-readable unit label (e.g. 'USD Billions', "
            "'Million people', 'Percentage')."
        )}],
        max_completion_tokens=60,
    )
    try:
        raw = resp.choices[0].message.content.strip().strip("```json").strip("```")
        u   = json.loads(raw)
        u.setdefault("prefix", ""); u.setdefault("suffix", ""); u.setdefault("description", "")
        return u
    except Exception:
        return {"prefix": "", "suffix": "", "description": ""}

def fmt(val: float, u: dict) -> str:
    p, sf, av = u.get("prefix", ""), u.get("suffix", ""), abs(val)
    if av >= 5e11:  return f"{p}{val/1e12:.2f}T{sf}"
    if av >= 5e8:   return f"{p}{val/1e9:.2f}B{sf}"
    if av >= 5e5:   return f"{p}{val/1e6:.2f}M{sf}"
    if av >= 5e2:   return f"{p}{val/1e3:.1f}K{sf}"
    if av >= 1:     return f"{p}{val:.1f}{sf}"
    return f"{p}{val:.2f}{sf}"

# ── Caption ────────────────────────────────────────────────────────────────────
def generate_caption(topic: str, chart_title: str,
                     df: pd.DataFrame, units: dict) -> tuple[str, list[str]]:
    cols  = list(df.columns)
    years = list(df.index)
    lines = [f"Period: {years[0]}–{years[-1]}, unit: {units.get('description', 'values')}"]
    for col in cols:
        s, e  = df[col].iloc[0], df[col].iloc[-1]
        pct   = ((e - s) / s * 100) if s != 0 else 0
        lines.append(f"  {col}: {fmt(s, units)} → {fmt(e, units)} ({pct:+.1f}%)")
    summary = "\n".join(lines)

    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": (
            f"You are a data scientist creating an Instagram reel about '{topic}'.\n"
            f"Chart title: '{chart_title}'\nData summary:\n{summary}\n\n"
            "Write an Instagram CAPTION (3-5 sentences): explain the key trend, a notable "
            "inflection point, growth rates, a comparative insight — use actual numbers. "
            "End with one engaging question.\n\n"
            "Then write HASHTAGS: exactly 10 tags. MUST include: #fyp #viral #trending "
            "#dataviz #datavisualization — then 5 topic-specific ones. Space-separated.\n\n"
            "Format:\nCAPTION: <text>\nHASHTAGS: #tag1 #tag2 ..."
        )}],
        max_completion_tokens=400,
    )
    text     = resp.choices[0].message.content.strip()
    caption  = ""
    hashtags = []
    for line in text.splitlines():
        ls = line.strip()
        if ls.startswith("CAPTION:"):
            caption = ls[len("CAPTION:"):].strip()
        elif ls.startswith("HASHTAGS:"):
            raw = ls[len("HASHTAGS:"):].strip().split()
            hashtags = [t.lstrip("#") for t in raw if t.startswith("#")][:10]
    if not caption and "CAPTION:" in text:
        caption = text.split("CAPTION:")[1].split("HASHTAGS:")[0].strip()
    if not hashtags:
        hashtags = ["fyp", "viral", "trending", "dataviz", "datavisualization"]
    return caption, hashtags

# ── Animation ──────────────────────────────────────────────────────────────────
def create_line_race_video(
    df: pd.DataFrame,
    chart_title: str,
    icon_arrays: list[np.ndarray],
    units: dict,
    steps_per_period: int,
    raw_path: str,
) -> str:

    n_periods = len(df)
    n_lines   = min(len(df.columns), len(LINE_COLORS))
    df        = df.iloc[:, :n_lines]
    cols      = list(df.columns)
    colors    = LINE_COLORS[:n_lines]

    total_frames = (n_periods - 1) * steps_per_period + 1
    x_raw        = np.arange(n_periods, dtype=float)
    x_interp     = np.linspace(0, n_periods - 1, total_frames)
    y_interp     = {c: np.interp(x_interp, x_raw, df[c].values.astype(float))
                    for c in cols}
    idx_labels   = [str(v) for v in df.index]

    FIG_W, FIG_H, DPI = 6.75, 12.0, 160

    # ── Axis limits (from actual data) ────────────────────────────────────────
    all_y    = np.concatenate([y_interp[c] for c in cols])
    y_min, y_max = float(all_y.min()), float(all_y.max())
    y_range  = y_max - y_min if y_max != y_min else 1.0
    y_pad    = y_range * 0.10
    xlim_lo  = -0.35
    xlim_hi  = n_periods - 0.65
    ylim_lo  = y_min - y_pad
    ylim_hi  = y_max + y_pad * 3.0
    x_range  = xlim_hi - xlim_lo
    y_total  = ylim_hi - ylim_lo

    # Icon size in data coordinates (square appearance in video)
    ax_w_px  = (AX_R - AX_L) * FIG_W * DPI
    ax_h_px  = (AX_T - AX_B) * FIG_H * DPI
    dx       = ICON_PX / ax_w_px * x_range
    dy       = ICON_PX / ax_h_px * y_total

    # ── Figure ────────────────────────────────────────────────────────────────
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG, dpi=DPI)
    gs  = fig.add_gridspec(
        2, 1, height_ratios=[0.10, 0.90],
        left=AX_L, right=AX_R, top=AX_T, bottom=AX_B, hspace=0.02,
    )
    title_ax = fig.add_subplot(gs[0])
    ax       = fig.add_subplot(gs[1])

    for a in (title_ax, ax):
        a.set_facecolor(BG)
        a.patch.set_facecolor(BG)

    # ── Title strip ───────────────────────────────────────────────────────────
    title_ax.axis("off")
    title_ax.text(0.5, 0.70, chart_title,
                  transform=title_ax.transAxes,
                  color="#FFFFFF", fontsize=16, fontweight="bold",
                  ha="center", va="center")
    title_ax.text(0.5, 0.18, "↓  Read caption to know more  ↓",
                  transform=title_ax.transAxes,
                  color="#666666", fontsize=8, ha="center", va="center",
                  fontstyle="italic")

    # ── Chart styling ─────────────────────────────────────────────────────────
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#333333")
        ax.spines[sp].set_linewidth(0.8)
    ax.yaxis.grid(True, color="#1A1A1A", linewidth=0.8, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # ── Axes limits from DATA ─────────────────────────────────────────────────
    ax.set_xlim(xlim_lo, xlim_hi)
    ax.set_ylim(ylim_lo, ylim_hi)

    # X-axis ticks — actual period labels
    tick_step = max(1, n_periods // 6)
    tick_pos  = list(range(0, n_periods, tick_step))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([idx_labels[i] for i in tick_pos],
                       color="#CCCCCC", fontsize=9)
    ax.tick_params(axis="x", colors="#CCCCCC", labelsize=9)

    # Y-axis — actual values, formatted with units
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt(v, units)))
    ax.tick_params(axis="y", labelcolor="#CCCCCC", labelsize=9)

    # ── Unit guide text ───────────────────────────────────────────────────────
    unit_desc = units.get("description", "")
    if unit_desc:
        ax.text(0.01, 0.995, f"Values in {unit_desc}",
                transform=ax.transAxes,
                color="#888888", fontsize=8, ha="left", va="top",
                fontstyle="italic")

    # ── Per-series artists ────────────────────────────────────────────────────
    outer_glow, inner_glow, main_lines, val_labels, icon_ims = [], [], [], [], []

    for i, col in enumerate(cols):
        c = colors[i]

        # Multi-layer glow
        og, = ax.plot([], [], color=c, linewidth=18, alpha=0.07,
                      solid_capstyle="round", zorder=2)
        ig, = ax.plot([], [], color=c, linewidth=8,  alpha=0.18,
                      solid_capstyle="round", zorder=3)
        ml, = ax.plot([], [], color=c, linewidth=2.4,
                      solid_capstyle="round", zorder=4)

        # Icon via imshow — starts off-screen left, moves to data position
        im = ax.imshow(icon_arrays[i],
                       extent=[-99, -99 + dx, ylim_lo, ylim_lo + dy],
                       zorder=10, clip_on=False, aspect="auto",
                       interpolation="bilinear")

        # Value label: "Col: value"
        lbl = ax.text(0, 0, "", color=c,
                      fontsize=9, fontweight="bold",
                      va="center", ha="left", zorder=12, clip_on=False)

        outer_glow.append(og)
        inner_glow.append(ig)
        main_lines.append(ml)
        icon_ims.append(im)
        val_labels.append(lbl)

    # ── Legend ────────────────────────────────────────────────────────────────
    ax.legend(
        handles=[plt.Line2D([0], [0], color=colors[i], linewidth=2.2, label=cols[i])
                 for i in range(n_lines)],
        loc="upper left",
        frameon=True, framealpha=0.20,
        facecolor="#0A0A0A", edgecolor="#333333",
        labelcolor="#FFFFFF", fontsize=9,
        handlelength=1.4, borderpad=0.6, labelspacing=0.4,
    )

    # ── Period counter ────────────────────────────────────────────────────────
    period_txt = ax.text(0.65, 0.05, "",
                         transform=ax.transAxes,
                         color="#FFFFFF", fontsize=34, fontweight="bold",
                         ha="right", va="bottom", alpha=0.85)

    # ── Branding ──────────────────────────────────────────────────────────────
    fig.text(0.5, 0.018, BRAND, ha="center", va="bottom",
             fontsize=8, color="#2E2E2E")

    # ── Update function ───────────────────────────────────────────────────────
    def update(frame: int):
        f     = min(frame, total_frames - 1)
        x_now = x_interp[:f + 1]
        p_idx = int(np.clip(round(x_interp[f]), 0, n_periods - 1))

        for i, col in enumerate(cols):
            y_now = y_interp[col][:f + 1]
            outer_glow[i].set_data(x_now, y_now)
            inner_glow[i].set_data(x_now, y_now)
            main_lines[i].set_data(x_now, y_now)

            if len(x_now) > 0:
                xend, yend = float(x_now[-1]), float(y_now[-1])
                # Move icon along the line — imshow set_extent
                icon_ims[i].set_extent([
                    xend - dx / 2, xend + dx / 2,
                    yend - dy / 2, yend + dy / 2,
                ])
                # Value label to the right of icon
                val_labels[i].set_position((xend + dx / 2 + 0.15, yend))
                val_labels[i].set_text(f"{col}: {fmt(yend, units)}")
            else:
                icon_ims[i].set_extent([-99, -99 + dx, ylim_lo, ylim_lo + dy])
                val_labels[i].set_text("")

        period_txt.set_text(idx_labels[p_idx])

    ani = mpl_animation.FuncAnimation(
        fig, update, frames=total_frames,
        interval=1000 / FPS, blit=False,
    )
    writer = mpl_animation.FFMpegWriter(
        fps=FPS, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "17"],
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
                p = os.path.join(tmp_dir, "music.mp3")
                open(p, "wb").write(r.content)
                return p
        except Exception:
            pass
    return None

def post_produce(raw_path: str, final_path: str, tmp_dir: str) -> str:
    from moviepy import VideoFileClip, AudioFileClip, concatenate_audioclips
    video    = VideoFileClip(raw_path)
    duration = video.duration
    mp = download_music(tmp_dir)
    if mp:
        try:
            audio = AudioFileClip(mp)
            if audio.duration < duration:
                audio = concatenate_audioclips([audio] * int(np.ceil(duration / audio.duration)))
            audio = audio.subclipped(0, duration).with_volume_scaled(0.32)
            video = video.with_audio(audio)
        except Exception:
            pass
    video.write_videofile(
        final_path, fps=FPS, codec="libx264", audio_codec="aac",
        temp_audiofile=os.path.join(tmp_dir, "tmp_audio.m4a"),
        remove_temp=True, logger=None, preset="fast",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "17"],
    )
    video.close()
    return final_path

# ── Trending topics ────────────────────────────────────────────────────────────
def _reddit(sub: str, n: int = 5) -> list[str]:
    try:
        r = requests.get(f"https://www.reddit.com/r/{sub}/hot.json?limit={n}",
                         headers=REDDIT_UA, timeout=8)
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
        raw.extend(_reddit(sub))
        if len(raw) >= 20:
            break
    if not raw:
        return FALLBACK_TOPICS
    block = "\n".join(f"- {t}" for t in raw[:25])
    try:
        resp = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content":
                    "Turn trending headlines into 8 animated line-chart prompts (time comparisons). "
                    "One per line, no bullets, no numbering."},
                {"role": "user", "content": f"Trending:\n{block}\n\nGenerate 8 chart prompts."},
            ],
            max_completion_tokens=300,
        )
        lines = [l.strip() for l in resp.choices[0].message.content.strip().splitlines()
                 if l.strip()]
        if len(lines) >= 4:
            return lines[:8]
    except Exception:
        pass
    return FALLBACK_TOPICS

# ── Streamlit UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Topic-to-Reel", page_icon="🎬", layout="centered")
st.title("Topic-to-Reel")
st.caption("Generate a 1080×1920 animated line chart Reel — from a topic, file, or your own data.")

# ── Session state init ────────────────────────────────────────────────────────
for key, default in [
    ("topic_value", ""),
    ("pending_df",  None),
    ("pending_title", ""),
    ("pending_topic", ""),
    ("grid_df", pd.DataFrame({
        "Year":     [2020, 2021, 2022, 2023, 2024],
        "Series A": [100.0, 120.0, 145.0, 175.0, 210.0],
        "Series B": [80.0,  92.0, 108.0, 127.0, 150.0],
    })),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Data input tabs ───────────────────────────────────────────────────────────
tab_ai, tab_upload, tab_paste, tab_grid = st.tabs([
    "🤖  AI Topic", "📁  Upload File", "📋  Paste Data", "📊  Grid Editor",
])

# ── Tab 1: AI Topic ───────────────────────────────────────────────────────────
with tab_ai:
    with st.spinner("Fetching trending topics…"):
        suggestions = get_trending_topics()

    st.markdown("**Trending today:**")
    bcols = st.columns(2)
    for idx, sug in enumerate(suggestions):
        if bcols[idx % 2].button(sug, key=f"sug_{idx}", use_container_width=True):
            st.session_state["topic_value"] = sug
            st.rerun()

    topic_in = st.text_area(
        "Or type your own topic",
        key="topic_value",
        placeholder="e.g.  US vs China GDP 1980–2024",
        height=80,
    )
    if st.button("Load AI Data", key="load_ai", type="primary", use_container_width=True):
        if not topic_in.strip():
            st.warning("Enter a topic first.")
        else:
            with st.spinner("Researching data…"):
                df, title = extract_data_from_llm(topic_in)
                st.session_state.update(
                    pending_df=df, pending_title=title, pending_topic=topic_in,
                )
            st.success(f"Loaded: **{title}**  ({len(df)} rows × {len(df.columns)} series)")

# ── Tab 2: Upload ─────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown("Upload a **CSV** or **XLSX** file. First column = time index, rest = series.")
    uploaded = st.file_uploader("Choose file", type=["csv", "xlsx", "xls", "txt"],
                                key="file_upload")
    if uploaded:
        try:
            df, title = parse_uploaded_file(uploaded)
            st.session_state.update(
                pending_df=df, pending_title=title, pending_topic=title,
            )
            st.success(f"Loaded: **{title}**  ({len(df)} rows × {len(df.columns)} series)")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to parse file: {e}")

# ── Tab 3: Paste raw data ─────────────────────────────────────────────────────
with tab_paste:
    st.markdown("Paste **any structured or semi-structured data** — CSV, Wikipedia table, "
                "rough notes. AI will extract it.")
    raw_paste = st.text_area("Paste data here", height=220,
                             placeholder="Year,India,USA\n2000,477,10.3T\n2005,820,13T\n...")
    if st.button("Parse & Load", key="parse_paste", use_container_width=True):
        if not raw_paste.strip():
            st.warning("Paste some data first.")
        else:
            with st.spinner("Parsing with AI…"):
                try:
                    df, title = parse_pasted_data(raw_paste)
                    st.session_state.update(
                        pending_df=df, pending_title=title, pending_topic=title,
                    )
                    st.success(f"Loaded: **{title}**  ({len(df)} rows × {len(df.columns)} series)")
                    st.dataframe(df.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Parsing failed: {e}")

# ── Tab 4: Grid editor ────────────────────────────────────────────────────────
with tab_grid:
    st.markdown("Edit the table directly. **First column = time index**, rest = series.  "
                "Use the ➕ at the bottom to add rows.")

    # Add column
    with st.expander("Add a new column"):
        c1, c2 = st.columns([3, 1])
        new_col = c1.text_input("Column name", key="new_col_name", label_visibility="collapsed",
                                placeholder="New series name…")
        if c2.button("Add", key="add_col") and new_col.strip():
            st.session_state["grid_df"][new_col.strip()] = 0.0
            st.rerun()

    edited = st.data_editor(
        st.session_state["grid_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor_widget",
    )
    chart_title_input = st.text_input("Chart title (optional)",
                                      placeholder="e.g. Global GDP Race")

    if st.button("Use Grid Data", key="use_grid", use_container_width=True):
        try:
            df = edited.copy()
            df = df.set_index(df.columns[0])
            df = df.apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0)
            title = chart_title_input.strip() or "Custom Data Reel"
            st.session_state.update(
                pending_df=df, pending_title=title, pending_topic=title,
                grid_df=edited,
            )
            st.success(f"Grid loaded: **{title}**  ({len(df)} rows × {len(df.columns)} series)")
        except Exception as e:
            st.error(f"Grid error: {e}")

# ── Data preview ──────────────────────────────────────────────────────────────
if st.session_state["pending_df"] is not None:
    df_preview = st.session_state["pending_df"]
    st.divider()
    with st.expander(f"📊 Preview: **{st.session_state['pending_title']}** "
                     f"({len(df_preview)} rows × {len(df_preview.columns)} series)", expanded=False):
        st.dataframe(df_preview, use_container_width=True)

# ── Settings ──────────────────────────────────────────────────────────────────
st.divider()
with st.expander("⚙️  Settings", expanded=True):
    steps = st.slider(
        "Animation speed  (frames per time period)",
        min_value=8, max_value=60, value=30, step=4,
        help="Lower = faster video. Higher = slower, more dramatic.",
    )
    n_p = len(st.session_state["pending_df"]) if st.session_state["pending_df"] is not None else 25
    est_sec = (n_p - 1) * steps / FPS
    st.caption(
        f"⏱ At this speed: each period = **{steps/FPS:.1f}s** → "
        f"estimated video length **{est_sec:.0f}s** "
        f"({est_sec/60:.1f} min)  |  {n_p} time periods detected"
    )

# ── Generate ──────────────────────────────────────────────────────────────────
st.divider()
generate = st.button("🎬  Generate Reel", type="primary", use_container_width=True,
                     disabled=(st.session_state["pending_df"] is None))
if st.session_state["pending_df"] is None:
    st.info("Load data in any tab above, then hit **Generate Reel**.")

if generate and st.session_state["pending_df"] is not None:
    df          = st.session_state["pending_df"]
    chart_title = st.session_state["pending_title"]
    topic_str   = st.session_state["pending_topic"] or chart_title

    progress = st.progress(0, text="Starting…")
    status   = st.empty()

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_mp4   = os.path.join(tmp_dir, "race.mp4")
            final_mp4 = os.path.join(tmp_dir, "reel.mp4")

            progress.progress(5, text="Detecting units…")
            status.info("📐  Detecting data units…")
            n_lines     = min(len(df.columns), len(LINE_COLORS))
            df          = df.iloc[:, :n_lines]
            cols_used   = list(df.columns)
            colors_used = LINE_COLORS[:n_lines]
            units       = detect_units(topic_str, df)

            progress.progress(14, text="Fetching icons…")
            status.info("🏳  Fetching category icons…")
            icon_arrays = get_icons(cols_used, colors_used)

            progress.progress(22, text="Rendering animation…")
            status.info(f"📈  Rendering: **{chart_title}**  "
                        f"({len(df)} periods × {n_lines} series, speed={steps})")
            create_line_race_video(df, chart_title, icon_arrays, units, steps, raw_mp4)

            progress.progress(78, text="Adding music…")
            status.info("🎬  Adding background music…")
            post_produce(raw_mp4, final_mp4, tmp_dir)

            progress.progress(91, text="Generating caption…")
            status.info("✍️  Writing data-scientist caption…")
            caption, hashtags = generate_caption(topic_str, chart_title, df, units)

            progress.progress(100, text="Done!")
            status.success("✅  Your Reel is ready!")

            with open(final_mp4, "rb") as f:
                video_bytes = f.read()

        # ── Output ────────────────────────────────────────────────────────────
        st.video(video_bytes)
        st.download_button(
            "⬇️  Download MP4 (1080 × 1920)", video_bytes,
            file_name=f"reel_{int(time.time())}.mp4", mime="video/mp4",
            use_container_width=True,
        )

        st.subheader("Caption  —  copy & paste to Instagram")
        hashtag_line = " ".join(f"#{h}" for h in hashtags)
        st.code(f"{caption}\n\n{hashtag_line}", language=None)

    except Exception as exc:
        progress.empty()
        status.empty()
        st.error(f"Something went wrong: {exc}")
        with st.expander("Error details"):
            st.exception(exc)
