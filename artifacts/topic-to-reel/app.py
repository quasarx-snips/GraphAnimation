"""
Topic-to-Reel — 1080×1920 animated Reel generator (line + bar chart modes).
"""
from __future__ import annotations
import re, shutil, io, json, os, tempfile, time, traceback
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
_ff = shutil.which("ffmpeg")
if _ff:
    matplotlib.rcParams["animation.ffmpeg_path"] = _ff

import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import requests
import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import CubicSpline

# ── OpenAI (Replit AI Integrations) ───────────────────────────────────────────
client = OpenAI(
    base_url=os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL"),
    api_key=os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY"),
)

# ── Constants ──────────────────────────────────────────────────────────────────
FONT_BOLD = "/run/current-system/sw/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LINE_COLORS = [
    "#FF6B35", "#4FC3F7", "#69F0AE", "#FFD740",
    "#E040FB", "#FF7043", "#40C4FF", "#B9F6CA",
]
MUSIC_URLS = [
    "https://cdn.pixabay.com/download/audio/2021/09/23/audio_57bc8dcb4e.mp3",
    "https://cdn.pixabay.com/download/audio/2022/10/25/audio_d0bba6d4e2.mp3",
]
REDDIT_UA  = {"User-Agent": "TopicToReel/1.0"}
BRAND      = "worldstats.visualised"
BG         = "#000000"
FPS        = 30

FALLBACK_TOPICS = [
    "US vs China GDP 1980–2024",
    "Global CO₂ by continent 1990–2023",
    "EV sales by country 2015–2024",
    "Netflix vs YouTube vs TikTok subscribers 2015–2024",
    "iPhone vs Android market share 2010–2024",
    "Global renewable vs fossil energy 2000–2023",
    "Top social media platforms by users 2012–2024",
    "India vs USA vs China population 2000–2024",
]

_COL_SUFFIXES = {
    "pop", "population", "gdp", "val", "value", "pct", "percent", "perc",
    "bn", "tn", "mn", "share", "total", "count", "num", "number", "rate",
    "growth", "index", "idx", "score", "data", "stat", "stats", "avg",
    "mean", "sum", "abs", "rel", "annual", "yearly", "monthly", "daily",
}

# ── Column-name cleaning ───────────────────────────────────────────────────────
def _clean_col_name(name: str) -> str:
    parts = re.split(r"[_\-\s]+", str(name).strip())
    kept  = [p for p in parts if p and p.lower() not in _COL_SUFFIXES]
    return " ".join(p.capitalize() for p in (kept or parts[:1]))

# ── Custom exceptions ──────────────────────────────────────────────────────────
class DataIndexError(ValueError): pass
class ResamplingFrequencyMismatch(ValueError): pass
class RenderingError(RuntimeError): pass

# ── Font helper ────────────────────────────────────────────────────────────────
def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_BOLD, size)
    except Exception:
        return ImageFont.load_default()

# ── Icon helpers ───────────────────────────────────────────────────────────────
def _to_circle(pil: Image.Image, size: int = 48) -> np.ndarray:
    img  = pil.convert("RGBA").resize((size, size), Image.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
    img.putalpha(mask)
    return np.array(img)

def _initials(color_hex: str, label: str, size: int = 48) -> np.ndarray:
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    r = int(color_hex[1:3], 16); g = int(color_hex[3:5], 16); b = int(color_hex[5:7], 16)
    draw.ellipse((0, 0, size-1, size-1), fill=(r, g, b, 210))
    letter = label[0].upper()
    font   = _font(max(14, size // 2))
    bb     = draw.textbbox((0, 0), letter, font=font)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    draw.text(((size-tw)//2 - bb[0], (size-th)//2 - bb[1]),
              letter, fill=(255, 255, 255, 255), font=font)
    return np.array(img)

def _make_glow(color_hex: str, size: int = 64) -> np.ndarray:
    rv = int(color_hex[1:3], 16) / 255.0
    gv = int(color_hex[3:5], 16) / 255.0
    bv = int(color_hex[5:7], 16) / 255.0
    img = np.zeros((size, size, 4), dtype=np.float32)
    cx, cy = size / 2.0, size / 2.0
    Y_g, X_g = np.mgrid[0:size, 0:size]
    dist  = np.sqrt((X_g - cx)**2 + (Y_g - cy)**2)
    alpha = np.clip(1.0 - dist / (size / 2.0), 0.0, 1.0) ** 1.6 * 0.55
    img[..., 0] = rv; img[..., 1] = gv; img[..., 2] = bv; img[..., 3] = alpha
    return (img * 255).astype(np.uint8)

def _flag(code: str, size: int = 48) -> np.ndarray | None:
    try:
        r = requests.get(f"https://flagcdn.com/48x36/{code.lower()}.png", timeout=6)
        if r.status_code == 200:
            return _to_circle(Image.open(io.BytesIO(r.content)), size)
    except Exception:
        pass
    return None

def _identify_countries(cols: list[str]) -> dict[str, str | None]:
    try:
        resp = client.chat.completions.create(
            model="gpt-5.1",
            messages=[{"role": "user", "content": (
                f"For each name {cols}, if it is a country return ISO 3166-1 alpha-2 "
                "code (lowercase), else null. ONLY JSON. "
                'Example: {"India":"in","USA":"us","GDP":null}'
            )}],
            max_completion_tokens=150,
        )
        raw = resp.choices[0].message.content.strip().strip("```json").strip("```")
        return json.loads(raw)
    except Exception:
        return {c: None for c in cols}

def get_icons(cols: list[str], colors: list[str],
              size: int = 48, custom_icons: dict | None = None) -> list[np.ndarray]:
    if custom_icons is None:
        custom_icons = {}
    cmap = _identify_countries(cols)
    out  = []
    for i, col in enumerate(cols):
        if col in custom_icons:
            try:
                out.append(_to_circle(Image.open(io.BytesIO(custom_icons[col])), size))
                continue
            except Exception:
                pass
        code = cmap.get(col)
        icon = _flag(code, size) if code else None
        out.append(icon if icon is not None else _initials(colors[i], col, size))
    return out

# ── Date index helpers ─────────────────────────────────────────────────────────
def _try_parse_dates(idx: list) -> tuple[pd.DatetimeIndex, list[str]] | None:
    """Try to interpret index as dates; return (dates, display_labels) or None."""
    try:
        dates = pd.to_datetime([str(v).strip() for v in idx], infer_datetime_format=True)
        if dates.isnull().any():
            return None
        delta_days = (dates[-1] - dates[0]).days
        if delta_days <= 0:
            return None
        if delta_days <= 90:      # daily
            labels = [d.strftime("%-d %b %Y") for d in dates]
        elif delta_days <= 900:   # monthly
            labels = [d.strftime("%b %Y") for d in dates]
        else:                     # yearly
            labels = [str(d.year) for d in dates]
        return dates, labels
    except Exception:
        return None

def _format_period_label(raw_label: str) -> str:
    """Format a period label nicely for the large counter display."""
    try:
        d = pd.to_datetime(raw_label, infer_datetime_format=True)
        if "day" in raw_label or len(raw_label) > 7:
            return d.strftime("%-d %B %Y")
        return d.strftime("%B %Y")
    except Exception:
        return raw_label

# ── Data parsing ───────────────────────────────────────────────────────────────
def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if not df.index.name or df.index.name == "":
        df = df.set_index(df.columns[0])
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.ffill().bfill().fillna(0)
    if len(df) < 2:
        raise DataIndexError("Need at least 2 rows (time periods).")
    df.columns = [_clean_col_name(c) for c in df.columns]
    return df

def extract_data_from_llm(topic: str) -> tuple[pd.DataFrame, str]:
    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": "Return ONLY clean CSV. No markdown."},
            {"role": "user", "content": (
                f'Topic: "{topic}"\n'
                "Rules:\n"
                "- First column = time index. For annual data use integer years. "
                "For monthly data use YYYY-MM-DD format (1st of each month). "
                "For daily data use YYYY-MM-DD format.\n"
                "- 2–6 category columns, short noun-only names (e.g. 'India' not 'India_pop')\n"
                "- Raw numeric values only, 15–30 rows, realistic trends.\n"
                "Return ONLY CSV, no markdown."
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
        messages=[{"role": "user", "content":
                   f"Short chart title (≤7 words, title case) for: {topic}. Return only the title."}],
        max_completion_tokens=30,
    )
    return df, t.choices[0].message.content.strip().strip("\"'")

def parse_uploaded_file(file) -> tuple[pd.DataFrame, str]:
    name = file.name.lower()
    if name.endswith((".csv", ".txt")):
        df = pd.read_csv(file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    else:
        raise DataIndexError(f"Unsupported file type: {file.name}")
    df    = _clean_df(df)
    title = file.name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
    return df, title

def parse_pasted_data(raw_text: str) -> tuple[pd.DataFrame, str]:
    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": "Return ONLY clean CSV. No markdown."},
            {"role": "user", "content": (
                "Convert this data into a clean CSV:\n"
                "- First column: time index (use YYYY-MM-DD for dates)\n"
                "- Remaining: short noun-only column names, numeric values only\n\n"
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
        messages=[{"role": "user", "content":
                   f"Short chart title (≤7 words) for columns {list(df.columns)}. Return only title."}],
        max_completion_tokens=30,
    )
    return df, t.choices[0].message.content.strip().strip("\"'")

def parse_csv_text(csv_text: str, chart_title: str) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(io.StringIO(csv_text.strip()))
    df = _clean_df(df)
    return df, chart_title.strip() or "Custom Reel"

# ── Temporal resampling (date-aware) ──────────────────────────────────────────
def temporal_resample(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    idx = df.index.tolist()

    # 1. Try date parsing first
    date_result = _try_parse_dates(idx)
    if date_result is not None:
        _, labels = date_result
        return df, labels

    # 2. Numeric fallback
    try:
        nums = [float(str(v).strip()) for v in idx]
    except (ValueError, TypeError):
        return df, [str(v) for v in idx]

    delta = nums[-1] - nums[0]
    if delta > 1:
        return df, [str(v) for v in idx]
    elif 0 < delta <= 1:
        n_weeks   = max(4, round(delta * 52) + 1)
        x_orig    = np.linspace(0, 1, len(df))
        x_fine    = np.linspace(0, 1, n_weeks)
        start_y   = int(nums[0])
        labels    = [f"Week {w+1}, {start_y}" for w in range(n_weeks)]
        new_data  = {}
        for col in df.columns:
            cs = CubicSpline(x_orig, df[col].values.astype(float))
            new_data[col] = np.clip(cs(x_fine), df[col].min() * 0.8, df[col].max() * 1.2)
        return pd.DataFrame(new_data, index=range(n_weeks)), labels
    else:
        raise ResamplingFrequencyMismatch(f"Non-monotonic index: delta={delta}")

# ── Units ──────────────────────────────────────────────────────────────────────
def detect_units(topic: str, df: pd.DataFrame) -> dict:
    sample = {c: {"first": float(df[c].iloc[0]), "last": float(df[c].iloc[-1])}
              for c in list(df.columns)[:3]}
    try:
        resp = client.chat.completions.create(
            model="gpt-5.1",
            messages=[{"role": "user", "content": (
                f'Topic: "{topic}", sample: {json.dumps(sample)}.\n'
                'Return ONLY JSON: {"prefix":"$","suffix":"","description":"USD Billions"}'
            )}],
            max_completion_tokens=60,
        )
        raw = resp.choices[0].message.content.strip().strip("```json").strip("```")
        u   = json.loads(raw)
        u.setdefault("prefix", ""); u.setdefault("suffix", ""); u.setdefault("description", "")
        return u
    except Exception:
        return {"prefix": "", "suffix": "", "description": ""}

def fmt(val: float, u: dict) -> str:
    p, sf, av = u.get("prefix",""), u.get("suffix",""), abs(val)
    if av >= 5e11: return f"{p}{val/1e12:.2f}T{sf}"
    if av >= 5e8:  return f"{p}{val/1e9:.2f}B{sf}"
    if av >= 5e5:  return f"{p}{val/1e6:.2f}M{sf}"
    if av >= 5e2:  return f"{p}{val/1e3:.1f}K{sf}"
    if av >= 1:    return f"{p}{val:.1f}{sf}"
    return f"{p}{val:.2f}{sf}"

# ── Caption ────────────────────────────────────────────────────────────────────
def generate_caption(topic: str, chart_title: str,
                     df: pd.DataFrame, units: dict) -> tuple[str, list[str]]:
    cols  = list(df.columns)
    years = list(df.index)
    lines = [f"Period: {years[0]}–{years[-1]}, unit: {units.get('description','values')}"]
    for col in cols:
        s, e = df[col].iloc[0], df[col].iloc[-1]
        pct  = ((e-s)/s*100) if s != 0 else 0
        lines.append(f"  {col}: {fmt(s,units)} → {fmt(e,units)} ({pct:+.1f}%)")
    try:
        resp = client.chat.completions.create(
            model="gpt-5.1",
            messages=[{"role": "user", "content": (
                f"You are a data scientist making an Instagram Reel about '{topic}'.\n"
                f"Chart: '{chart_title}'\nData:\n" + "\n".join(lines) + "\n\n"
                "Write a CAPTION (3-5 sentences) with key trends, cite numbers. End with a question.\n"
                "Then HASHTAGS: exactly 10 including #fyp #viral #trending #dataviz "
                "#datavisualization + 5 topic-specific. Space-separated.\n"
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
            hashtags = ["fyp","viral","trending","dataviz","datavisualization"]
        return caption, hashtags
    except Exception:
        return "Check out this data trend!", ["fyp","viral","trending","dataviz","datavisualization"]

# ── Collision avoidance ────────────────────────────────────────────────────────
def avoid_collisions(positions: list[float], min_gap: float,
                     iterations: int = 60) -> list[float]:
    n = len(positions)
    if n <= 1:
        return list(positions)
    order = sorted(range(n), key=lambda i: positions[i])
    pos   = np.array([positions[order[i]] for i in range(n)], dtype=float)
    for _ in range(iterations):
        moved = False
        for i in range(n-1):
            gap = pos[i+1] - pos[i]
            if gap < min_gap:
                push = (min_gap - gap) / 2.0 + 1e-10
                pos[i] -= push; pos[i+1] += push
                moved = True
        if not moved:
            break
    result = [0.0] * n
    for si, oi in enumerate(order):
        result[oi] = float(pos[si])
    return result

# ── Shared figure scaffold ─────────────────────────────────────────────────────
# Layout constants (figure fractions from bottom):
#   0.00–0.03  bottom pad
#   0.03–0.19  period counter (HUGE)
#   0.19–0.74  chart area
#   0.74–0.88  title + subtitle block
#   0.88–1.00  brand + padding
_AX_B = 0.20; _AX_T = 0.74
_AX_L = 0.13; _AX_R = 0.70
FIG_W, FIG_H, DPI = 6.75, 12.0, 160

def _make_figure(chart_title: str, subtitle: str) -> tuple:
    """Create figure with shared layout; return (fig, ax, period_txt)."""
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG, dpi=DPI)

    # Brand – very top, tiny, dim
    fig.text(0.50, 0.972, BRAND, ha="center", va="top",
             fontsize=8, color="#3A3A3A", fontstyle="italic")

    # Title – large bold, left-aligned
    fig.text(_AX_L, 0.955, chart_title, ha="left", va="top",
             fontsize=20, fontweight="bold", color="#FFFFFF",
             wrap=True)

    # Subtitle (unit description) – smaller, dimmer
    if subtitle:
        fig.text(_AX_L, 0.905, subtitle, ha="left", va="top",
                 fontsize=10, color="#888888")

    # Chart axes
    ax = fig.add_axes([_AX_L, _AX_B, _AX_R - _AX_L, _AX_T - _AX_B])
    ax.set_facecolor(BG)

    # Spines: only left + bottom, rest hidden
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_color("#444444")
    ax.spines["bottom"].set_linewidth(0.8)

    # No gridlines
    ax.grid(False)
    ax.set_axisbelow(True)

    # Period counter – HUGE, centred below chart
    period_txt = fig.text(0.50, 0.115, "",
                          ha="center", va="center",
                          fontsize=34, fontweight="bold", color="#FFFFFF")

    return fig, ax, period_txt

# ── Line race video ────────────────────────────────────────────────────────────
def create_line_race_video(
    df: pd.DataFrame,
    idx_labels: list[str],
    chart_title: str,
    icon_arrays: list[np.ndarray],
    units: dict,
    steps_per_period: int,
    raw_path: str,
) -> str:
    n_periods = len(df)
    n_lines   = min(len(df.columns), len(LINE_COLORS))
    df        = df.iloc[:, :n_lines].copy()
    cols      = list(df.columns)
    colors    = LINE_COLORS[:n_lines]
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    total_frames = (n_periods-1) * steps_per_period + 1
    x_raw   = np.arange(n_periods, dtype=float)
    x_dense = np.linspace(0, n_periods-1, total_frames)

    y_interp: dict[str, np.ndarray] = {}
    for col in cols:
        raw_vals = df[col].values.astype(float)
        cs       = CubicSpline(x_raw, raw_vals)
        y_interp[col] = np.clip(cs(x_dense),
                                raw_vals.min() * 0.90,
                                raw_vals.max() * 1.10)
    Y_MAT = np.column_stack([y_interp[c] for c in cols])

    y_min_data = float(Y_MAT.min())
    y_max_data = float(Y_MAT.max())
    y_range    = max(y_max_data - y_min_data, 1.0)
    y_pad      = y_range * 0.08

    use_log = (y_min_data > 0) and ((y_max_data / y_min_data) > 10)
    if use_log:
        ylim_lo = y_min_data * 0.5; ylim_hi = y_max_data * 2.0
    else:
        ylim_lo = y_min_data - y_pad; ylim_hi = y_max_data + y_pad * 2.5
    y_total = ylim_hi - ylim_lo

    zoom_frames = 0 if use_log else min(int(3.0 * FPS), total_frames // 3)

    xlim_lo       = -0.35
    xlim_hi_max   = n_periods - 0.65
    ELASTIC_AHEAD = 1.65

    # Icon display sizes in points (1 point = 1/72 inch; DPI conversion handled by matplotlib)
    # AnnotationBbox / OffsetImage renders at fixed display pixels regardless of data scale.
    ICON_DP      = 22   # icon rendered size in display points
    ICON_ZOOM    = ICON_DP / 48.0           # 48px source → 22pt display
    GLOW_ZOOM    = (ICON_DP * 2.0) / 80.0  # 80px source → 44pt display
    # Pixel offset (in display points) from line-tip dot to left edge of icon
    DOT_TO_ICON  = 7
    # Pixel offset from left edge of icon to left edge of text label
    ICON_TO_LBL  = ICON_DP + 5

    unit_desc = units.get("description", "")
    fig, ax, period_txt = _make_figure(chart_title, unit_desc)

    if use_log:
        ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt(v, units)))
    ax.tick_params(axis="y", labelcolor="#666666", labelsize=8, length=0, pad=4)

    tick_step = max(1, n_periods // 6)
    tick_pos  = list(range(0, n_periods, tick_step))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([idx_labels[i] for i in tick_pos],
                       color="#555555", fontsize=8, rotation=0)
    ax.tick_params(axis="x", length=0, pad=4)

    ax.set_xlim(xlim_lo, min(xlim_lo + ELASTIC_AHEAD + 1.0, xlim_hi_max))
    ax.set_ylim(ylim_lo, ylim_hi)

    # Per-series artists
    main_lines  = []
    halo_outer  = []   # large soft glow ring
    halo_inner  = []   # medium glow ring
    tip_dots    = []   # crisp dot at line tip
    icon_boxes  = []   # AnnotationBbox wrapping flag/initial icon
    val_labels  = []

    for i, col in enumerate(cols):
        c = colors[i]
        ml, = ax.plot([], [], color=c, linewidth=2.2,
                      solid_capstyle="round", zorder=4)

        # Layered glow: outer ring → inner ring → solid dot
        ho, = ax.plot([], [], "o", color=c, markersize=28, alpha=0.10,
                      zorder=6, clip_on=False)
        hi, = ax.plot([], [], "o", color=c, markersize=16, alpha=0.22,
                      zorder=7, clip_on=False)
        dot,= ax.plot([], [], "o", color=c, markersize=7,
                      markerfacecolor=c, markeredgecolor="#FFFFFF",
                      markeredgewidth=1.4, zorder=9, clip_on=False)

        # Icon via AnnotationBbox — pixel-accurate, no data-coord math
        iim = OffsetImage(icon_arrays[i], zoom=ICON_ZOOM, interpolation="lanczos")
        iab = AnnotationBbox(
            iim, (0, 0), xycoords="data",
            box_alignment=(0, 0.5),   # left-centre aligned at anchor point
            frameon=False, zorder=10,
            clip_on=False,
        )
        iab.set_visible(False)
        ax.add_artist(iab)

        lbl = ax.text(0, 0, "",
                      color=c, fontsize=8.5, fontweight="bold",
                      va="center", ha="left", zorder=12, clip_on=False,
                      multialignment="left", linespacing=1.4)

        main_lines.append(ml)
        halo_outer.append(ho)
        halo_inner.append(hi)
        tip_dots.append(dot)
        icon_boxes.append(iab)
        val_labels.append(lbl)

    # Cached transform — recalculated when axes limits change
    _tf_cache: dict = {}

    def _data_to_disp(x: float, y: float) -> tuple[float, float]:
        return ax.transData.transform((x, y))

    def _disp_to_data(xd: float, yd: float) -> tuple[float, float]:
        return ax.transData.inverted().transform((xd, yd))

    def update(frame: int):
        f         = min(frame, total_frames-1)
        current_x = float(x_dense[f])
        x_now_arr = x_dense[:f+1]
        p_idx     = int(np.clip(round(current_x), 0, n_periods-1))

        xlim_right = min(current_x + ELASTIC_AHEAD, xlim_hi_max)
        ax.set_xlim(xlim_lo, xlim_right)

        if not use_log and zoom_frames > 0 and f < zoom_frames:
            t_ease  = 1.0 - (1.0 - f/zoom_frames) ** 3
            zoom    = 2.0 - t_ease
            center  = float(Y_MAT[f].mean())
            half_h  = y_total / (2.0 * zoom)
            new_ylo = max(ylim_lo, center - half_h)
            new_yhi = min(ylim_hi, center + half_h)
            ax.set_ylim(new_ylo, new_yhi)
        else:
            if not use_log:
                ax.set_ylim(ylim_lo, ylim_hi)

        cur_ylim = ax.get_ylim()

        # Compute a "1 data-unit on y" → display-pixels ratio for nudge gap
        try:
            _, y0_px = _data_to_disp(0, cur_ylim[0])
            _, y1_px = _data_to_disp(0, cur_ylim[1])
            px_per_y = abs(y1_px - y0_px)
            ylim_span = cur_ylim[1] - cur_ylim[0]
            # Minimum gap in data units = ICON_DP * 1.15 display points
            min_gap_data = ICON_DP * 1.15 / px_per_y * ylim_span if px_per_y > 0 else ylim_span * 0.05
        except Exception:
            min_gap_data = (cur_ylim[1] - cur_ylim[0]) * 0.05

        raw_ypos = [float(Y_MAT[f, i]) for i in range(n_lines)]
        nudged   = avoid_collisions(raw_ypos, min_gap_data)

        for i, col in enumerate(cols):
            y_arr = y_interp[col][:f+1]
            main_lines[i].set_data(x_now_arr, y_arr)

            if len(x_now_arr) > 0:
                xend = float(x_now_arr[-1])
                yend = float(y_arr[-1])
                ny   = float(np.clip(nudged[i],
                                     cur_ylim[0] + ylim_span * 0.02,
                                     cur_ylim[1] - ylim_span * 0.02))

                # Layered glow at actual line tip
                halo_outer[i].set_data([xend], [yend])
                halo_inner[i].set_data([xend], [yend])
                tip_dots[i].set_data([xend], [yend])

                # Convert tip to display px, offset icon to the right
                try:
                    tx, ty = _data_to_disp(xend, ny)
                    ix, _  = _disp_to_data(tx + DOT_TO_ICON, ty)
                    lx, _  = _disp_to_data(tx + DOT_TO_ICON + ICON_TO_LBL, ty)
                except Exception:
                    ix, lx = xend, xend

                icon_boxes[i].xy = (ix, ny)
                icon_boxes[i].set_visible(True)

                val_labels[i].set_position((lx, ny))
                val_labels[i].set_text(f"{col}\n{fmt(float(Y_MAT[f, i]), units)}")
            else:
                halo_outer[i].set_data([], [])
                halo_inner[i].set_data([], [])
                tip_dots[i].set_data([], [])
                icon_boxes[i].set_visible(False)
                val_labels[i].set_text("")

        period_txt.set_text(_format_period_label(idx_labels[p_idx]))

    try:
        ani = mpl_animation.FuncAnimation(
            fig, update, frames=total_frames,
            interval=1000/FPS, blit=False,
        )
        writer = mpl_animation.FFMpegWriter(
            fps=FPS, codec="libx264",
            extra_args=["-pix_fmt","yuv420p","-preset","fast","-crf","17",
                        "-vf","pad=ceil(iw/2)*2:ceil(ih/2)*2"],
        )
        ani.save(raw_path, writer=writer, dpi=DPI,
                 savefig_kwargs={"facecolor": BG})
    except Exception as exc:
        raise RenderingError(f"Line animation export failed: {exc}") from exc
    finally:
        plt.close(fig)

    return raw_path

# ── Bar race video ─────────────────────────────────────────────────────────────
def create_bar_race_video(
    df: pd.DataFrame,
    idx_labels: list[str],
    chart_title: str,
    icon_arrays: list[np.ndarray],
    units: dict,
    total_duration_secs: float,
    raw_path: str,
) -> str:
    n_periods = len(df)
    n_bars    = min(len(df.columns), len(LINE_COLORS))
    df        = df.iloc[:, :n_bars].copy()
    cols      = list(df.columns)
    colors    = LINE_COLORS[:n_bars]

    steps_per_period = max(2, int(total_duration_secs * FPS / max(n_periods-1, 1)))
    total_frames     = (n_periods-1) * steps_per_period + 1

    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    x_raw   = np.arange(n_periods, dtype=float)
    x_dense = np.linspace(0, n_periods-1, total_frames)

    y_interp: dict[str, np.ndarray] = {}
    for col in cols:
        raw_vals = df[col].values.astype(float)
        cs       = CubicSpline(x_raw, raw_vals)
        y_interp[col] = np.clip(cs(x_dense), 0.0, raw_vals.max() * 1.05)

    Y_MAT  = np.column_stack([y_interp[c] for c in cols])
    x_max  = float(Y_MAT.max()) * 1.08 or 1.0
    BAR_H  = 0.30    # thin, sharp bars
    LERP   = 0.14    # per-frame interpolation speed for rank transitions
    # Icon display size (display points) — sits centred on the bar
    ICON_DP_B = 20
    ICON_ZM_B = ICON_DP_B / 48.0

    unit_desc = units.get("description", "")
    fig, ax, period_txt = _make_figure(chart_title, unit_desc)

    # All spines off; we draw a thin baseline at x=0 ourselves
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_yticks([])
    ax.xaxis.set_visible(False)

    # x range: include a left label strip (negative x = label area)
    LABEL_STRIP = x_max * 0.30   # data units reserved on the left for labels
    ax.set_xlim(-LABEL_STRIP, x_max * 1.04)
    ax.set_ylim(-0.65, n_bars - 0.35)
    ax.invert_yaxis()   # rank-0 (highest) at the top

    # Thin baseline
    ax.axvline(x=0, color="#333333", linewidth=0.6, zorder=1)

    # ── Initial ranking (sort descending by first-frame value) ─────────────────
    init_vals    = {col: float(Y_MAT[0, i]) for i, col in enumerate(cols)}
    sorted_init  = sorted(cols, key=lambda c: -init_vals[c])
    # bar_ypos[col] = current animated y position (rank 0 = top due to invert)
    bar_ypos     = {col: float(r) for r, col in enumerate(sorted_init)}

    # ── Bar patches (plain Rectangle — sharp edges) ────────────────────────────
    bar_patches: list[mpatches.Rectangle] = []
    for i, col in enumerate(cols):
        y0 = bar_ypos[col]
        rect = mpatches.Rectangle(
            (0, y0 - BAR_H / 2), 0.001, BAR_H,
            linewidth=0, facecolor=colors[i], zorder=4,
        )
        ax.add_patch(rect)
        bar_patches.append(rect)

    # ── Icons (AnnotationBbox) — centred on bar, inside the label strip ────────
    icon_boxes_b: list[AnnotationBbox] = []
    for i, col in enumerate(cols):
        y0  = bar_ypos[col]
        iim = OffsetImage(icon_arrays[i], zoom=ICON_ZM_B, interpolation="lanczos")
        iab = AnnotationBbox(
            iim, (-LABEL_STRIP * 0.18, y0),
            xycoords="data",
            box_alignment=(0.5, 0.5),
            frameon=False, zorder=10, clip_on=False,
        )
        ax.add_artist(iab)
        icon_boxes_b.append(iab)

    # ── Name labels (left of icon) ─────────────────────────────────────────────
    name_labels: list = []
    for i, col in enumerate(cols):
        y0  = bar_ypos[col]
        lbl = ax.text(
            -LABEL_STRIP * 0.33, y0, col,
            color=colors[i], fontsize=8.5, fontweight="bold",
            va="center", ha="right", zorder=12, clip_on=False,
        )
        name_labels.append(lbl)

    # ── Value labels at right tip of each bar ─────────────────────────────────
    val_labels: list = []
    for i in range(n_bars):
        y0  = list(bar_ypos.values())[i]
        lbl = ax.text(
            0.001, y0, "",
            color="#FFFFFF", fontsize=8.5, fontweight="bold",
            va="center", ha="left", zorder=12, clip_on=False,
        )
        val_labels.append(lbl)

    def update_bar(frame: int):
        f     = min(frame, total_frames-1)
        p_idx = int(np.clip(round(x_dense[f]), 0, n_periods-1))

        # Current interpolated values
        cur_vals = {col: float(Y_MAT[f, ci]) for ci, col in enumerate(cols)}

        # Target ranking: highest value → rank 0 (top)
        sorted_cols  = sorted(cols, key=lambda c: -cur_vals[c])
        target_ranks = {col: float(r) for r, col in enumerate(sorted_cols)}

        # Smooth position lerp — gradual overtake animation
        for col in cols:
            bar_ypos[col] += (target_ranks[col] - bar_ypos[col]) * LERP

        # Update each bar, icon, and label using the lerped position
        for i, col in enumerate(cols):
            y  = bar_ypos[col]
            v  = cur_vals[col]
            vw = max(v, 0.001)

            bar_patches[i].set_y(y - BAR_H / 2)
            bar_patches[i].set_width(vw)

            icon_boxes_b[i].xy = (-LABEL_STRIP * 0.18, y)
            name_labels[i].set_y(y)
            val_labels[i].set_y(y)
            val_labels[i].set_x(v + x_max * 0.012)
            val_labels[i].set_text(fmt(v, units))

        period_txt.set_text(_format_period_label(idx_labels[p_idx]))

    try:
        ani = mpl_animation.FuncAnimation(
            fig, update_bar, frames=total_frames,
            interval=1000/FPS, blit=False,
        )
        writer = mpl_animation.FFMpegWriter(
            fps=FPS, codec="libx264",
            extra_args=["-pix_fmt","yuv420p","-preset","fast","-crf","17",
                        "-vf","pad=ceil(iw/2)*2:ceil(ih/2)*2"],
        )
        ani.save(raw_path, writer=writer, dpi=DPI,
                 savefig_kwargs={"facecolor": BG})
    except Exception as exc:
        raise RenderingError(f"Bar animation export failed: {exc}") from exc
    finally:
        plt.close(fig)

    return raw_path

# ── Music + post-production ────────────────────────────────────────────────────
def _download_music(tmp_dir: str) -> str | None:
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
    mp = _download_music(tmp_dir)
    if mp:
        try:
            audio = AudioFileClip(mp)
            if audio.duration < duration:
                n     = int(np.ceil(duration / audio.duration))
                audio = concatenate_audioclips([audio] * n)
            audio = audio.subclipped(0, duration).with_volume_scaled(0.28)
            video = video.with_audio(audio)
        except Exception:
            pass
    video.write_videofile(
        final_path, fps=FPS, codec="libx264", audio_codec="aac",
        temp_audiofile=os.path.join(tmp_dir, "tmp_audio.m4a"),
        remove_temp=True, logger=None, preset="fast",
        ffmpeg_params=["-pix_fmt","yuv420p","-crf","17"],
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
    for sub in ["dataisbeautiful","worldnews","science","technology"]:
        raw.extend(_reddit(sub))
        if len(raw) >= 20:
            break
    if not raw:
        return FALLBACK_TOPICS
    try:
        resp = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content":
                    "Turn headlines into 8 animated line-chart prompts (time-series). "
                    "One per line, no bullets, no numbering."},
                {"role": "user", "content":
                    f"Trending:\n" + "\n".join(f"- {t}" for t in raw[:24]) +
                    "\n\nGenerate 8 chart prompts."},
            ],
            max_completion_tokens=300,
        )
        lines = [l.strip() for l in
                 resp.choices[0].message.content.strip().splitlines() if l.strip()]
        if len(lines) >= 4:
            return lines[:8]
    except Exception:
        pass
    return FALLBACK_TOPICS

# ── Streamlit UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Topic-to-Reel", page_icon="🎬", layout="centered")

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("Topic-to-Reel")
    st.caption("Generate a 1080×1920 animated Reel from a topic, file, or your own data.")
with col_h2:
    st.markdown(
        f"<div style='text-align:right;padding-top:28px;color:#555;"
        f"font-style:italic;font-size:13px'>{BRAND}</div>",
        unsafe_allow_html=True,
    )

# ── Session state ──────────────────────────────────────────────────────────────
_defaults: dict = {
    "topic_value":    "",
    "pending_df":     None,
    "pending_title":  "",
    "pending_topic":  "",
    "pending_labels": None,
    "last_video":     None,
    "last_caption":   "",
    "last_hashtags":  [],
    "history":        [],
    "custom_icons":   {},
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def _store_df(df: pd.DataFrame, title: str, topic: str) -> None:
    try:
        df_rs, labels = temporal_resample(df)
    except ResamplingFrequencyMismatch:
        df_rs, labels = df, [str(v) for v in df.index]
    st.session_state.update(
        pending_df=df_rs, pending_title=title,
        pending_topic=topic, pending_labels=labels,
        custom_icons={},
    )

# ── Input tabs ─────────────────────────────────────────────────────────────────
tab_ai, tab_upload, tab_paste, tab_csv = st.tabs([
    "🤖  AI Topic", "📁  Upload File", "📋  Paste Data", "📝  CSV Input",
])

# Tab 1 — AI Topic
with tab_ai:
    with st.spinner("Fetching trending topics…"):
        suggestions = get_trending_topics()
    st.markdown("**Trending today:**")
    bcols = st.columns(2)
    for idx, sug in enumerate(suggestions):
        if bcols[idx % 2].button(sug, key=f"sug_{idx}", use_container_width=True):
            st.session_state["topic_value"] = sug
            st.rerun()

    topic_in = st.text_area("Or type your own topic", key="topic_value",
                             placeholder="e.g. US vs China GDP 1980–2024", height=80)
    if st.button("Load AI Data", key="load_ai", type="primary", use_container_width=True):
        if not topic_in.strip():
            st.warning("Enter a topic first.")
        else:
            with st.spinner("Researching data with AI…"):
                try:
                    df, title = extract_data_from_llm(topic_in)
                    _store_df(df, title, topic_in)
                    pdf = st.session_state["pending_df"]
                    st.success(f"Loaded **{title}** — {len(pdf)} rows × {len(pdf.columns)} series")
                    st.dataframe(pdf.head(8), use_container_width=True)
                except DataIndexError as e:
                    st.error(f"**DataIndexError:** {e}")
                except Exception as e:
                    st.error(f"**Error ({type(e).__name__}):** {e}")

# Tab 2 — Upload
with tab_upload:
    st.markdown("Upload a **CSV** or **XLSX**. First column = time index, rest = series.")
    uploaded = st.file_uploader("Choose file", type=["csv","xlsx","xls","txt"],
                                key="file_upload")
    if uploaded:
        try:
            df, title = parse_uploaded_file(uploaded)
            _store_df(df, title, title)
            pdf = st.session_state["pending_df"]
            st.success(f"Loaded **{title}** — {len(pdf)} rows × {len(pdf.columns)} series")
            st.dataframe(pdf.head(8), use_container_width=True)
        except (DataIndexError, ResamplingFrequencyMismatch) as e:
            st.error(f"**{type(e).__name__}:** {e}")
        except Exception as e:
            st.error(f"**Error ({type(e).__name__}):** {e}")

# Tab 3 — Paste (AI-parsed)
with tab_paste:
    st.markdown("Paste **any data** — Wikipedia table, rough notes. AI structures it into CSV.")
    raw_paste = st.text_area("Paste data here", height=200,
                             placeholder="Year,India,USA\n2000,477,10300\n2005,820,13000\n…")
    if st.button("Parse & Load", key="parse_paste", use_container_width=True):
        if not raw_paste.strip():
            st.warning("Paste some data first.")
        else:
            with st.spinner("Parsing with AI…"):
                try:
                    df, title = parse_pasted_data(raw_paste)
                    _store_df(df, title, title)
                    pdf = st.session_state["pending_df"]
                    st.success(f"Loaded **{title}** — {len(pdf)} rows × {len(pdf.columns)} series")
                    st.dataframe(pdf.head(8), use_container_width=True)
                except (DataIndexError, ResamplingFrequencyMismatch) as e:
                    st.error(f"**{type(e).__name__}:** {e}")
                except Exception as e:
                    st.error(f"**Error ({type(e).__name__}):** {e}")

# Tab 4 — Raw CSV input
with tab_csv:
    st.markdown(
        "Paste **raw CSV** — first column is the time index, "
        "remaining columns are your data series."
    )
    st.markdown(
        """
        ```
        Year,India,China,USA
        2000,477,1211,10300
        2005,820,1810,13000
        2010,1676,5879,14964
        2015,2103,11016,18037
        2020,2660,14688,20894
        2024,3732,17795,27360
        ```
        """
    )
    csv_input = st.text_area("Paste CSV here", height=220,
                             placeholder="Year,Series A,Series B\n2000,100,80\n2001,110,85\n…",
                             key="csv_raw_input")
    csv_title = st.text_input("Chart title", placeholder="e.g. GDP Race 2000–2024",
                               key="csv_title_input")
    if st.button("Load CSV", key="load_csv", use_container_width=True):
        if not csv_input.strip():
            st.warning("Paste CSV data first.")
        else:
            try:
                df, title = parse_csv_text(csv_input, csv_title)
                _store_df(df, title, title)
                pdf = st.session_state["pending_df"]
                st.success(f"Loaded **{title}** — {len(pdf)} rows × {len(pdf.columns)} series")
                st.dataframe(pdf.head(10), use_container_width=True)
            except (DataIndexError, ResamplingFrequencyMismatch) as e:
                st.error(f"**{type(e).__name__}:** {e}")
            except Exception as e:
                st.error(f"**Error ({type(e).__name__}):** {e}")

# ── Data preview ──────────────────────────────────────────────────────────────
if st.session_state["pending_df"] is not None:
    pdf = st.session_state["pending_df"]
    st.divider()
    with st.expander(
        f"📊 **{st.session_state['pending_title']}** — "
        f"{len(pdf)} rows × {len(pdf.columns)} series", expanded=False
    ):
        st.dataframe(pdf, use_container_width=True)

    # Custom icon uploads
    with st.expander("🖼️  Custom icons  (optional — leave blank for auto flags/initials)",
                     expanded=False):
        series_cols = list(pdf.columns)
        n_ic_cols   = min(len(series_cols), 4)
        ic_cols     = st.columns(n_ic_cols)
        for i, col in enumerate(series_cols):
            with ic_cols[i % n_ic_cols]:
                up = st.file_uploader(col, type=["png","jpg","jpeg","webp"],
                                      key=f"icon_{col}")
                if up is not None:
                    st.session_state["custom_icons"][col] = up.read()
                if col in st.session_state["custom_icons"]:
                    try:
                        st.image(Image.open(io.BytesIO(
                            st.session_state["custom_icons"][col])).resize((40,40)),
                            caption=f"{col} ✓", width=40)
                    except Exception:
                        pass
                    if st.button("✕ Remove", key=f"rm_{col}"):
                        del st.session_state["custom_icons"][col]
                        st.rerun()

# ── Settings ──────────────────────────────────────────────────────────────────
st.divider()
with st.expander("⚙️  Settings", expanded=True):
    chart_type = st.radio(
        "Chart type",
        ["📈  Line Chart Race", "📊  Bar Chart Race"],
        horizontal=True,
    )
    is_bar = "Bar" in chart_type

    if is_bar:
        bar_duration = st.slider(
            "Total animation duration (seconds)",
            min_value=5, max_value=60, value=20, step=5,
        )
        st.caption(
            f"⏱  {bar_duration}s total · bars animate through all "
            f"{len(st.session_state['pending_df']) if st.session_state['pending_df'] is not None else '—'} "
            "time periods"
        )
    else:
        steps = st.slider(
            "Frames per time period  (higher = slower, smoother)",
            min_value=8, max_value=60, value=28, step=4,
        )
        n_p = len(st.session_state["pending_df"]) if st.session_state["pending_df"] is not None else 25
        est = (n_p-1) * steps / FPS
        st.caption(
            f"⏱  Each period ≈ **{steps/FPS:.1f}s** → "
            f"estimated length **{est:.0f}s** ({est/60:.1f} min)  |  "
            f"{n_p} periods · Catmull-Rom spline"
        )

# ── Generate ──────────────────────────────────────────────────────────────────
st.divider()
generate = st.button(
    "🎬  Generate Reel", type="primary", use_container_width=True,
    disabled=(st.session_state["pending_df"] is None),
)
if st.session_state["pending_df"] is None:
    st.info("Load data in any tab above, then hit **Generate Reel**.")

if generate and st.session_state["pending_df"] is not None:
    df          = st.session_state["pending_df"]
    idx_labels  = st.session_state.get("pending_labels") or [str(v) for v in df.index]
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
            df_use      = df.iloc[:, :n_lines]
            colors_used = LINE_COLORS[:n_lines]
            units       = detect_units(topic_str, df_use)

            progress.progress(14, text="Fetching icons…")
            status.info("🏳  Fetching category icons…")
            icon_arrays = get_icons(
                list(df_use.columns), colors_used, size=48,
                custom_icons=st.session_state.get("custom_icons", {}),
            )

            progress.progress(22, text="Rendering animation…")
            if is_bar:
                status.info(
                    f"📊  Rendering bar race **{chart_title}**  —  "
                    f"{len(df_use)} periods × {n_lines} bars  ·  {bar_duration}s duration"
                )
                create_bar_race_video(
                    df_use, idx_labels, chart_title,
                    icon_arrays, units, bar_duration, raw_mp4,
                )
            else:
                n_frames = (len(df_use)-1) * steps + 1
                status.info(
                    f"📈  Rendering line race **{chart_title}**  —  "
                    f"{len(df_use)} periods × {n_lines} series  ·  {n_frames} frames"
                )
                create_line_race_video(
                    df_use, idx_labels, chart_title,
                    icon_arrays, units, steps, raw_mp4,
                )

            progress.progress(78, text="Adding music…")
            status.info("🎵  Adding background music…")
            post_produce(raw_mp4, final_mp4, tmp_dir)

            progress.progress(91, text="Generating caption…")
            status.info("✍️  Writing caption…")
            caption, hashtags = generate_caption(topic_str, chart_title, df_use, units)

            progress.progress(100, text="Done!")
            status.success("✅  Reel ready!")

            with open(final_mp4, "rb") as fh:
                video_bytes = fh.read()

        st.session_state["last_video"]    = video_bytes
        st.session_state["last_caption"]  = caption
        st.session_state["last_hashtags"] = hashtags

        st.session_state["history"].append({
            "ts":     datetime.now().strftime("%H:%M:%S"),
            "title":  chart_title,
            "type":   "bar" if is_bar else "line",
            "rows":   len(df_use),
            "series": n_lines,
            "bytes":  len(video_bytes),
        })
        st.session_state["history"] = st.session_state["history"][-5:]

    except (DataIndexError, ResamplingFrequencyMismatch, RenderingError) as exc:
        progress.empty(); status.empty()
        st.error(f"**{type(exc).__name__}:** {exc}")
        with st.expander("🔍 Debug trace"):
            st.code(traceback.format_exc())
    except Exception as exc:
        progress.empty(); status.empty()
        st.error(f"**Error ({type(exc).__name__}):** {exc}")
        with st.expander("🔍 Debug trace"):
            st.code(traceback.format_exc())

# ── Output ────────────────────────────────────────────────────────────────────
if st.session_state["last_video"]:
    video_bytes = st.session_state["last_video"]
    st.divider()
    st.subheader("Your Reel")
    st.video(video_bytes)

    st.download_button(
        label="⬇️  Download MP4  (1080 × 1920)",
        data=video_bytes,
        file_name=f"reel_{int(time.time())}.mp4",
        mime="video/mp4",
        use_container_width=True,
    )

    st.subheader("Caption — copy & paste to Instagram")
    hashtag_line = " ".join(f"#{h}" for h in st.session_state["last_hashtags"])
    st.code(f"{st.session_state['last_caption']}\n\n{hashtag_line}", language=None)

# ── History ───────────────────────────────────────────────────────────────────
if st.session_state["history"]:
    st.divider()
    with st.expander(
        f"🕘  Generation history  ({len(st.session_state['history'])} entries)",
        expanded=False
    ):
        for rec in reversed(st.session_state["history"]):
            icon = "📊" if rec.get("type") == "bar" else "📈"
            st.markdown(
                f"{icon} **{rec['ts']}** — {rec['title']}  "
                f"·  {rec['rows']} periods · {rec['series']} series  "
                f"·  {rec['bytes']/1e6:.1f} MB"
            )
