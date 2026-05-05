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
DEFAULT_COLORS = [
    "#FF6B35", "#4FC3F7", "#69F0AE", "#FFD740",
    "#E040FB", "#FF7043", "#40C4FF", "#B9F6CA",
]
MUSIC_URLS = [
    "https://cdn.pixabay.com/download/audio/2021/09/23/audio_57bc8dcb4e.mp3",
    "https://cdn.pixabay.com/download/audio/2022/10/25/audio_d0bba6d4e2.mp3",
]
REDDIT_UA  = {"User-Agent": "TopicToReel/1.0 (by /u/topictoreelbot)"}
BRAND      = "worldstats.visualised"
BG         = "#000000"
FPS        = 30

FALLBACK_TOPICS = [
    "US vs China GDP 2000–2024",
    "Global CO₂ by continent 2000–2023",
    "EV sales by country 2018–2024",
    "Netflix vs YouTube vs TikTok subscribers 2018–2024",
    "iPhone vs Android market share 2015–2024",
    "Global renewable vs fossil energy 2010–2024",
    "Top social media platforms by users 2015–2024",
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
    try:
        dates = pd.to_datetime([str(v).strip() for v in idx], infer_datetime_format=True)
        if dates.isnull().any():
            return None
        delta_days = (dates[-1] - dates[0]).days
        if delta_days <= 0:
            return None
        if delta_days <= 90:
            labels = [d.strftime("%-d %b %Y") for d in dates]
        elif delta_days <= 900:
            labels = [d.strftime("%b %Y") for d in dates]
        else:
            labels = [str(d.year) for d in dates]
        return dates, labels
    except Exception:
        return None

def _format_period_label(raw_label: str) -> str:
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
    date_result = _try_parse_dates(idx)
    if date_result is not None:
        _, labels = date_result
        return df, labels
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
    cols   = list(df.columns)
    sample = {c: {"first": float(df[c].iloc[0]), "last": float(df[c].iloc[-1])}
              for c in cols[:4]}
    try:
        resp = client.chat.completions.create(
            model="gpt-5.1",
            messages=[{"role": "user", "content": (
                f'Data topic: "{topic}"\n'
                f'Column names: {cols}\n'
                f'Sample values: {json.dumps(sample)}\n\n'
                'Identify the REAL measurement units for these values.\n'
                'Return ONLY valid JSON:\n'
                '{"prefix":"","suffix":"","description":"","is_pct":false}\n'
                'Examples:\n'
                '  GDP ($B) → {"prefix":"$","suffix":"B","description":"GDP (USD Billions)","is_pct":false}\n'
                '  Market share → {"prefix":"","suffix":"%","description":"Market share (%)","is_pct":true}\n'
                '  Population → {"prefix":"","suffix":"","description":"Population","is_pct":false}'
            )}],
            max_completion_tokens=120,
        )
        raw = resp.choices[0].message.content.strip().strip("```json").strip("```").strip()
        u   = json.loads(raw)
        u.setdefault("prefix", ""); u.setdefault("suffix", "")
        u.setdefault("description", ""); u.setdefault("is_pct", False)
        return u
    except Exception:
        return {"prefix": "", "suffix": "", "description": "", "is_pct": False}

def fmt(val: float, u: dict) -> str:
    p, sf   = u.get("prefix", ""), u.get("suffix", "")
    is_pct  = u.get("is_pct", False) or sf.strip() == "%"
    av      = abs(val)
    if is_pct:
        return f"{p}{val:.1f}{sf}"
    if av >= 5e11: return f"{p}{val/1e12:.2f}T{sf}"
    if av >= 5e8:  return f"{p}{val/1e9:.2f}B{sf}"
    if av >= 5e5:  return f"{p}{val/1e6:.2f}M{sf}"
    if av >= 2e3:  return f"{p}{val/1e3:.1f}K{sf}"
    if av >= 1:    return f"{p}{val:.1f}{sf}"
    return f"{p}{val:.3f}{sf}"

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

# ── Trend indicator ────────────────────────────────────────────────────────────
def _trend_arrow(cur: float, first: float) -> str:
    if cur > first * 1.001:
        return "▲"
    elif cur < first * 0.999:
        return "▼"
    return "●"

# ── Shared figure scaffold ─────────────────────────────────────────────────────
_AX_B = 0.17; _AX_T = 0.83
_AX_L = 0.13; _AX_R = 0.70
FIG_W, FIG_H, DPI = 6.75, 12.0, 160

def _make_figure(chart_title: str, subtitle: str, cta_text: str = "👇 Read caption for more") -> tuple:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG, dpi=DPI)

    # Brand — very top, tiny, dim
    fig.text(0.50, 0.977, BRAND, ha="center", va="top",
             fontsize=8, color="#3A3A3A", fontstyle="italic")

    # Title — large bold
    fig.text(_AX_L, 0.962, chart_title, ha="left", va="top",
             fontsize=18, fontweight="bold", color="#FFFFFF", wrap=True)

    # Subtitle (unit description)
    title_bottom = 0.920
    if subtitle:
        fig.text(_AX_L, title_bottom, subtitle, ha="left", va="top",
                 fontsize=10, color="#888888")
        title_bottom = 0.900

    # CTA hook — below subtitle, prominent, acts as a hook line
    fig.text(_AX_L, title_bottom - 0.004, cta_text, ha="left", va="top",
             fontsize=9, color="#FF6B35", fontstyle="italic")

    # Chart axes
    ax = fig.add_axes([_AX_L, _AX_B, _AX_R - _AX_L, _AX_T - _AX_B])
    ax.set_facecolor(BG)

    # Spines: only left + bottom
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_color("#333333")
    ax.spines["bottom"].set_linewidth(0.8)

    # Subtle horizontal gridlines
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", color="#1A1A1A", linewidth=0.5, linestyle="--", alpha=0.8)
    ax.set_axisbelow(True)

    # Period counter — HUGE, centred below chart
    period_txt = fig.text(0.50, 0.090, "",
                          ha="center", va="center",
                          fontsize=36, fontweight="bold", color="#FFFFFF")

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
    colors: list[str] | None = None,
    x_label: str = "",
    y_label: str = "",
) -> str:
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    n_periods = len(df)
    n_lines   = min(len(df.columns), 8)
    df        = df.iloc[:, :n_lines].copy()
    cols      = list(df.columns)
    line_colors = (colors or DEFAULT_COLORS)[:n_lines]

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

    # First-period values for trend arrow
    first_vals = [float(df[col].iloc[0]) for col in cols]

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

    zoom_frames   = 0 if use_log else min(int(3.0 * FPS), total_frames // 3)
    xlim_lo       = -0.35
    xlim_hi_max   = n_periods - 0.65
    ELASTIC_AHEAD = 1.65

    # Icon on the leading dot — larger and centered on tip
    ICON_DP     = 20   # display-point diameter of icon circle
    ICON_ZOOM   = ICON_DP / 48.0
    # Label offset from dot center
    LBL_OFFSET  = ICON_DP + 6
    LBL_FONT    = 8.5 if n_lines <= 4 else 7.5

    unit_desc = units.get("description", "")
    fig, ax, period_txt = _make_figure(chart_title, unit_desc)

    if x_label:
        ax.set_xlabel(x_label, color="#666666", fontsize=9, labelpad=6)
    if y_label:
        ax.set_ylabel(y_label, color="#666666", fontsize=9, labelpad=6)

    if use_log:
        ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt(v, units)))
    ax.tick_params(axis="y", labelcolor="#555555", labelsize=8, length=3, pad=4,
                   direction="out", width=0.5)

    n_ticks  = min(5, n_periods)
    tick_pos = ([int(round(j * (n_periods - 1) / (n_ticks - 1)))
                 for j in range(n_ticks)] if n_ticks > 1 else [0])
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([idx_labels[i] for i in tick_pos],
                       color="#555555", fontsize=8.5, rotation=0)
    ax.tick_params(axis="x", length=3, pad=5, direction="out", width=0.5)

    ax.set_xlim(xlim_lo, min(xlim_lo + ELASTIC_AHEAD + 1.0, xlim_hi_max))
    ax.set_ylim(ylim_lo, ylim_hi)

    main_lines  = []
    halo_outer  = []
    halo_inner  = []
    tip_dots    = []
    icon_boxes  = []
    val_labels  = []

    for i, col in enumerate(cols):
        c = line_colors[i]
        ml, = ax.plot([], [], color=c, linewidth=2.5,
                      solid_capstyle="round", zorder=4)

        ho, = ax.plot([], [], "o", color=c, markersize=16, alpha=0.10,
                      zorder=6, clip_on=False)
        hi, = ax.plot([], [], "o", color=c, markersize=10, alpha=0.25,
                      zorder=7, clip_on=False)
        dot,= ax.plot([], [], "o", color=c, markersize=6,
                      markerfacecolor=c, markeredgecolor="#FFFFFF",
                      markeredgewidth=1.2, zorder=9, clip_on=False)

        # Icon CENTERED on the dot tip (box_alignment 0.5,0.5)
        iim = OffsetImage(icon_arrays[i], zoom=ICON_ZOOM, interpolation="lanczos")
        iab = AnnotationBbox(
            iim, (0, 0), xycoords="data",
            box_alignment=(0.5, 0.5),
            frameon=False, zorder=11, clip_on=False,
        )
        iab.set_visible(False)
        ax.add_artist(iab)

        lbl = ax.text(0, 0, "",
                      color=c, fontsize=LBL_FONT, fontweight="bold",
                      va="center", ha="left", zorder=12, clip_on=False,
                      multialignment="left", linespacing=1.25)

        main_lines.append(ml)
        halo_outer.append(ho)
        halo_inner.append(hi)
        tip_dots.append(dot)
        icon_boxes.append(iab)
        val_labels.append(lbl)

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
            ax.set_ylim(max(ylim_lo, center - half_h),
                        min(ylim_hi, center + half_h))
        else:
            if not use_log:
                ax.set_ylim(ylim_lo, ylim_hi)

        cur_ylim  = ax.get_ylim()
        ylim_span = cur_ylim[1] - cur_ylim[0]

        try:
            _, y0_px = _data_to_disp(0, cur_ylim[0])
            _, y1_px = _data_to_disp(0, cur_ylim[1])
            px_per_y = abs(y1_px - y0_px)
            font_px      = LBL_FONT * DPI / 72.0 * 1.3
            n_rows       = 2 if n_lines <= 4 else 1
            min_gap_data = font_px * n_rows / px_per_y * ylim_span if px_per_y > 0 else ylim_span * 0.05
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

                halo_outer[i].set_data([xend], [yend])
                halo_inner[i].set_data([xend], [yend])
                tip_dots[i].set_data([xend], [yend])

                # Icon centered on the dot tip
                icon_boxes[i].xy = (xend, yend)
                icon_boxes[i].set_visible(True)

                # Label offset to the right of the icon
                try:
                    tx_tip, ty_tip = _data_to_disp(xend, yend)
                    # Offset label by icon radius + gap in display pixels
                    lx, _ = _disp_to_data(tx_tip + ICON_DP / 2 + LBL_OFFSET, ty_tip)
                    _, ly_nud = _data_to_disp(xend, ny)
                    lx_nud, _ = _disp_to_data(tx_tip + ICON_DP / 2 + LBL_OFFSET, ly_nud)
                except Exception:
                    lx_nud = xend + 0.1
                    ny_orig = ny

                cur_val = float(Y_MAT[f, i])
                arrow   = _trend_arrow(cur_val, first_vals[i])
                val_str = fmt(cur_val, units)
                if n_lines <= 4:
                    lbl_txt = f"{col}\n{arrow} {val_str}"
                else:
                    name_s  = col[:10] if len(col) > 10 else col
                    lbl_txt = f"{name_s}: {arrow} {val_str}"
                val_labels[i].set_position((lx_nud, ny))
                val_labels[i].set_text(lbl_txt)
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
    colors: list[str] | None = None,
    x_label: str = "",
    y_label: str = "",
) -> str:
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    n_periods = len(df)
    n_bars    = min(len(df.columns), 8)
    df        = df.iloc[:, :n_bars].copy()
    cols      = list(df.columns)
    bar_colors = (colors or DEFAULT_COLORS)[:n_bars]

    steps_per_period = max(2, int(total_duration_secs * FPS / max(n_periods-1, 1)))
    total_frames     = (n_periods-1) * steps_per_period + 1

    x_raw   = np.arange(n_periods, dtype=float)
    x_dense = np.linspace(0, n_periods-1, total_frames)

    y_interp: dict[str, np.ndarray] = {}
    for col in cols:
        raw_vals = df[col].values.astype(float)
        cs       = CubicSpline(x_raw, raw_vals)
        y_interp[col] = np.clip(cs(x_dense), 0.0, raw_vals.max() * 1.05)

    Y_MAT  = np.column_stack([y_interp[c] for c in cols])
    x_max  = float(Y_MAT.max()) * 1.08 or 1.0
    BAR_H  = 0.38
    LERP   = 0.14

    first_vals = {col: float(df[col].iloc[0]) for col in cols}

    unit_desc = units.get("description", "")
    fig, ax, period_txt = _make_figure(chart_title, unit_desc)

    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_yticks([])
    ax.xaxis.set_visible(False)

    LABEL_STRIP = x_max * 0.28
    ax.set_xlim(-LABEL_STRIP, x_max * 1.12)
    ax.set_ylim(-0.65, n_bars - 0.35)
    ax.invert_yaxis()
    ax.axvline(x=0, color="#2A2A2A", linewidth=0.8, zorder=1)

    init_vals   = {col: float(Y_MAT[0, i]) for i, col in enumerate(cols)}
    sorted_init = sorted(cols, key=lambda c: -init_vals[c])
    bar_ypos    = {col: float(r) for r, col in enumerate(sorted_init)}

    bar_patches: list[mpatches.Rectangle] = []
    for i, col in enumerate(cols):
        y0   = bar_ypos[col]
        rect = mpatches.Rectangle(
            (0, y0 - BAR_H / 2), 0.001, BAR_H,
            linewidth=0, facecolor=bar_colors[i], zorder=4, antialiased=False,
        )
        ax.add_patch(rect)
        bar_patches.append(rect)

    name_labels: list = []
    for i, col in enumerate(cols):
        y0  = bar_ypos[col]
        lbl = ax.text(
            -LABEL_STRIP * 0.12, y0, col,
            color=bar_colors[i], fontsize=8.5, fontweight="bold",
            va="center", ha="right", zorder=12, clip_on=False,
        )
        name_labels.append(lbl)

    val_labels: list = []
    for i, col in enumerate(cols):
        y0  = bar_ypos[col]
        lbl = ax.text(
            0.001, y0, "",
            color="#FFFFFF", fontsize=8.5, fontweight="bold",
            va="center", ha="left", zorder=12, clip_on=False,
        )
        val_labels.append(lbl)

    def update_bar(frame: int):
        f     = min(frame, total_frames-1)
        p_idx = int(np.clip(round(x_dense[f]), 0, n_periods-1))

        cur_vals = {col: float(Y_MAT[f, ci]) for ci, col in enumerate(cols)}
        sorted_cols  = sorted(cols, key=lambda c: -cur_vals[c])
        target_ranks = {col: float(r) for r, col in enumerate(sorted_cols)}

        for col in cols:
            bar_ypos[col] += (target_ranks[col] - bar_ypos[col]) * LERP

        for i, col in enumerate(cols):
            y  = bar_ypos[col]
            v  = cur_vals[col]
            vw = max(v, 0.001)

            bar_patches[i].set_y(y - BAR_H / 2)
            bar_patches[i].set_width(vw)

            name_labels[i].set_y(y)
            val_labels[i].set_y(y)
            val_labels[i].set_x(v + x_max * 0.014)

            arrow   = _trend_arrow(v, first_vals[col])
            val_labels[i].set_text(f"{arrow} {fmt(v, units)}")

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

# ── Static preview renderer ────────────────────────────────────────────────────
def render_preview_frame(
    df: pd.DataFrame,
    idx_labels: list[str],
    chart_title: str,
    units: dict,
    is_bar: bool = False,
    colors: list[str] | None = None,
    x_label: str = "",
    y_label: str = "",
) -> bytes:
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    cols     = list(df.columns)[:8]
    n_series = len(cols)
    line_colors = (colors or DEFAULT_COLORS)[:n_series]
    icon_arrs = [_initials(line_colors[i], cols[i], 48) for i in range(n_series)]
    unit_desc = units.get("description", "")
    first_vals = [float(df[col].iloc[0]) for col in cols]

    if is_bar:
        final_vals  = {col: float(df[col].iloc[-1]) for col in cols}
        sorted_cols = sorted(cols, key=lambda c: -final_vals[c])
        x_max       = max(final_vals.values()) * 1.08 or 1.0
        BAR_H       = 0.38
        LABEL_STRIP = x_max * 0.28

        fig, ax, period_txt = _make_figure(chart_title, unit_desc)
        period_txt.set_text(_format_period_label(idx_labels[-1]))

        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_yticks([]); ax.xaxis.set_visible(False)
        ax.set_xlim(-LABEL_STRIP, x_max * 1.12)
        ax.set_ylim(-0.65, n_series - 0.35)
        ax.invert_yaxis()
        ax.axvline(x=0, color="#2A2A2A", linewidth=0.8, zorder=1)

        for rank, col in enumerate(sorted_cols):
            v   = final_vals[col]
            ci  = cols.index(col)
            c   = line_colors[ci]
            ax.add_patch(mpatches.Rectangle(
                (0, rank - BAR_H / 2), max(v, 0.001), BAR_H,
                linewidth=0, facecolor=c, zorder=4, antialiased=False,
            ))
            ax.text(-LABEL_STRIP * 0.12, rank, col,
                    color=c, fontsize=9.5, fontweight="bold",
                    va="center", ha="right", clip_on=False)
            arrow = _trend_arrow(v, float(df[col].iloc[0]))
            ax.text(v + x_max * 0.014, rank, f"{arrow} {fmt(v, units)}",
                    color="#FFFFFF", fontsize=9.5, fontweight="bold",
                    va="center", ha="left", clip_on=False)
    else:
        n      = len(df)
        vals   = df[cols].values.astype(float)
        y_min  = vals.min(); y_max = vals.max()
        y_rng  = max(y_max - y_min, 1.0)
        y_pad  = y_rng * 0.08
        ylim_lo = y_min - y_pad
        ylim_hi = y_max + y_pad * 2.5
        fig, ax, period_txt = _make_figure(chart_title, unit_desc)
        period_txt.set_text(_format_period_label(idx_labels[-1]))

        if x_label:
            ax.set_xlabel(x_label, color="#666666", fontsize=9, labelpad=6)
        if y_label:
            ax.set_ylabel(y_label, color="#666666", fontsize=9, labelpad=6)

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt(v, units)))
        ax.tick_params(axis="y", labelcolor="#555555", labelsize=8, length=3, pad=4,
                       direction="out", width=0.5)
        ax.set_xlim(-0.35, n - 0.65)
        ax.set_ylim(ylim_lo, ylim_hi)

        ylim_span = ylim_hi - ylim_lo
        P_ICON_DP  = 20
        P_ICON_ZM  = P_ICON_DP / 48.0
        P_LBL_FONT = 8.5 if n_series <= 4 else 7.5
        n_rows_p   = 2 if n_series <= 4 else 1
        P_MIN_GAP  = (P_LBL_FONT * DPI / 72.0 * n_rows_p * 1.4) / (FIG_H * DPI) * ylim_span

        raw_ypos = [float(vals[-1, i]) for i in range(n_series)]
        nudged   = avoid_collisions(raw_ypos, P_MIN_GAP)

        n_ticks_p  = min(5, n)
        tick_pos_p = ([int(round(j * (n - 1) / (n_ticks_p - 1)))
                       for j in range(n_ticks_p)] if n_ticks_p > 1 else [0])
        ax.set_xticks(tick_pos_p)
        ax.set_xticklabels([idx_labels[i] for i in tick_pos_p],
                           color="#555555", fontsize=8.5)
        ax.tick_params(axis="x", length=3, pad=5, direction="out", width=0.5)

        cur_ylim  = (ylim_lo, ylim_hi)

        for i, col in enumerate(cols):
            c    = line_colors[i]
            x    = list(range(n))
            y    = vals[:, i]
            xend = float(n - 1)
            yend = float(y[-1])
            ny   = float(np.clip(nudged[i],
                                 cur_ylim[0] + ylim_span * 0.02,
                                 cur_ylim[1] - ylim_span * 0.02))

            ax.plot(x, y, color=c, linewidth=2.5, solid_capstyle="round", zorder=4)

            # Glow layers
            ax.plot(xend, yend, "o", color=c, markersize=16, alpha=0.10,
                    clip_on=False, zorder=6)
            ax.plot(xend, yend, "o", color=c, markersize=10, alpha=0.25,
                    clip_on=False, zorder=7)
            ax.plot(xend, yend, "o", color=c, markersize=6,
                    markeredgecolor="#FFFFFF", markeredgewidth=1.2,
                    clip_on=False, zorder=9)

            # Icon centered on dot
            iab = AnnotationBbox(
                OffsetImage(icon_arrs[i], zoom=P_ICON_ZM, interpolation="lanczos"),
                (xend, yend), xycoords="data",
                box_alignment=(0.5, 0.5),
                frameon=False, zorder=11, clip_on=False,
            )
            ax.add_artist(iab)

            # Label to the right of icon
            arrow   = _trend_arrow(yend, first_vals[i])
            val_s   = fmt(yend, units)
            lbl_txt = f"{col}\n{arrow} {val_s}" if n_series <= 4 else f"{col[:10]}: {arrow} {val_s}"

            # Approximate data-unit offset for label
            pts_to_data = ylim_span / ((_AX_T - _AX_B) * FIG_H * DPI / 72.0)
            lbl_x_off   = (P_ICON_DP / 2 + 6) * pts_to_data

            ax.text(xend + lbl_x_off, ny, lbl_txt,
                    color=c, fontsize=P_LBL_FONT, fontweight="bold",
                    va="center", ha="left", clip_on=False,
                    multialignment="left", linespacing=1.25)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ── Live trending topics ────────────────────────────────────────────────────────
def _reddit_headlines(n: int = 12) -> list[str]:
    titles = []
    for sub in ["dataisbeautiful", "worldnews", "science", "technology", "economics"]:
        try:
            r = requests.get(
                f"https://www.reddit.com/r/{sub}/hot.json?limit=6",
                headers=REDDIT_UA, timeout=6,
            )
            if r.status_code == 200:
                for p in r.json()["data"]["children"]:
                    if not p["data"].get("stickied"):
                        titles.append(p["data"]["title"])
        except Exception:
            pass
        if len(titles) >= n:
            break
    return titles[:n]

def _gnews_headlines(n: int = 10) -> list[str]:
    try:
        r = requests.get(
            "https://news.google.com/rss/search?q=statistics+data+economy+technology&hl=en-US&gl=US&ceid=US:en",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=8,
        )
        if r.status_code == 200:
            titles = re.findall(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", r.text)
            return [t for t in titles if t and "Google News" not in t and len(t) > 10][:n]
    except Exception:
        pass
    return []

@st.cache_data(ttl=1800, show_spinner=False)
def get_trending_topics() -> list[str]:
    raw: list[str] = []
    raw.extend(_reddit_headlines(12))
    raw.extend(_gnews_headlines(10))
    if not raw:
        return FALLBACK_TOPICS
    try:
        resp = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": (
                    "Turn these headlines into 8 animated data chart prompts suitable "
                    "for a line or bar chart race. Each prompt should cover a time range "
                    "ending in 2024 or 2025. One prompt per line, no bullets, no numbering."
                )},
                {"role": "user", "content":
                    "Recent headlines:\n" + "\n".join(f"- {t}" for t in raw[:20]) +
                    "\n\nGenerate 8 chart prompts."},
            ],
            max_completion_tokens=350,
        )
        lines = [l.strip() for l in
                 resp.choices[0].message.content.strip().splitlines() if l.strip()]
        if len(lines) >= 4:
            return lines[:8]
    except Exception:
        pass
    return FALLBACK_TOPICS

# ── Streamlit page config ──────────────────────────────────────────────────────
st.set_page_config(page_title="Topic-to-Reel", page_icon="🎬", layout="wide")

# ── Session state defaults ─────────────────────────────────────────────────────
_defaults: dict = {
    "topic_value":           "",
    "pending_df":            None,
    "pending_title":         "",
    "pending_topic":         "",
    "pending_labels":        None,
    "pending_units":         {},
    "pending_preview_bytes": None,
    "_preview_is_bar":       False,
    "last_video":            None,
    "last_caption":          "",
    "last_hashtags":         [],
    "history":               [],
    "custom_icons":          {},
    "custom_colors":         [],
    "custom_title":          "",
    "custom_subtitle":       "",
    "custom_x_label":        "",
    "custom_y_label":        "",
    "custom_unit_prefix":    "",
    "custom_unit_suffix":    "",
    "custom_series_names":   {},
    "input_mode":            "🤖  AI Topic",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def _store_df(df: pd.DataFrame, title: str, topic: str) -> None:
    try:
        df_rs, labels = temporal_resample(df)
    except ResamplingFrequencyMismatch:
        df_rs, labels = df, [str(v) for v in df.index]

    try:
        units = detect_units(topic, df_rs)
    except Exception:
        units = {"prefix": "", "suffix": "", "description": "", "is_pct": False}

    try:
        preview_bytes = render_preview_frame(df_rs, labels, title, units, is_bar=False)
    except Exception:
        preview_bytes = None

    st.session_state.update(
        pending_df=df_rs, pending_title=title,
        pending_topic=topic, pending_labels=labels,
        pending_units=units, pending_preview_bytes=preview_bytes,
        custom_icons={},
        custom_title=title,
        custom_subtitle=units.get("description", ""),
        custom_unit_prefix=units.get("prefix", ""),
        custom_unit_suffix=units.get("suffix", ""),
        custom_series_names={col: col for col in df_rs.columns},
        custom_colors=DEFAULT_COLORS[:len(df_rs.columns)],
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — All input + trending
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"### 🎬 Topic-to-Reel")
    st.caption(f"*{BRAND}*")
    st.divider()

    # ── Input mode ────────────────────────────────────────────────────────────
    input_mode = st.radio(
        "**Data source**",
        ["🤖  AI Topic", "📁  Upload File", "📋  Paste Data", "📝  Raw CSV"],
        index=["🤖  AI Topic", "📁  Upload File", "📋  Paste Data", "📝  Raw CSV"].index(
            st.session_state["input_mode"]
        ),
    )
    st.session_state["input_mode"] = input_mode
    st.divider()

    # ── AI Topic ──────────────────────────────────────────────────────────────
    if input_mode == "🤖  AI Topic":
        st.markdown("**🔥 Trending now**")
        col_r1, col_r2 = st.columns([3, 1])
        with col_r1:
            st.caption("From Reddit & news, refreshed every 30 min")
        with col_r2:
            if st.button("↺", help="Refresh trending topics", use_container_width=True):
                get_trending_topics.clear()
                st.rerun()

        with st.spinner("Fetching live topics…"):
            suggestions = get_trending_topics()

        for idx, sug in enumerate(suggestions):
            if st.button(sug, key=f"sug_{idx}", use_container_width=True):
                st.session_state["topic_value"] = sug
                st.rerun()

        st.divider()
        topic_in = st.text_area(
            "Or type your own topic",
            key="topic_value",
            placeholder="e.g. US vs China EV sales 2015–2025",
            height=80,
        )
        if st.button("🚀 Load AI Data", type="primary", use_container_width=True):
            if not topic_in.strip():
                st.warning("Enter a topic first.")
            else:
                with st.spinner("Researching with AI…"):
                    try:
                        df, title = extract_data_from_llm(topic_in)
                        _store_df(df, title, topic_in)
                        st.success(f"✅ Loaded **{title}**")
                    except DataIndexError as e:
                        st.error(f"**DataIndexError:** {e}")
                    except Exception as e:
                        st.error(f"**Error:** {e}")

    # ── Upload ────────────────────────────────────────────────────────────────
    elif input_mode == "📁  Upload File":
        st.markdown("Upload a **CSV** or **XLSX**.")
        st.caption("First column = time index, rest = series.")
        uploaded = st.file_uploader("Choose file", type=["csv","xlsx","xls","txt"],
                                    key="file_upload")
        if uploaded:
            try:
                df, title = parse_uploaded_file(uploaded)
                _store_df(df, title, title)
                st.success(f"✅ Loaded **{title}**")
            except (DataIndexError, ResamplingFrequencyMismatch) as e:
                st.error(f"**{type(e).__name__}:** {e}")
            except Exception as e:
                st.error(f"**Error:** {e}")

    # ── Paste ────────────────────────────────────────────────────────────────
    elif input_mode == "📋  Paste Data":
        st.markdown("Paste **any data** — Wikipedia table, rough notes.")
        st.caption("AI will structure it into a chart-ready CSV.")
        raw_paste = st.text_area("Paste data here", height=180,
                                 placeholder="Year,India,USA\n2000,477,10300\n…")
        if st.button("🔍 Parse & Load", use_container_width=True):
            if not raw_paste.strip():
                st.warning("Paste some data first.")
            else:
                with st.spinner("AI is parsing your data…"):
                    try:
                        df, title = parse_pasted_data(raw_paste)
                        _store_df(df, title, title)
                        st.success(f"✅ Loaded **{title}**")
                    except (DataIndexError, ResamplingFrequencyMismatch) as e:
                        st.error(f"**{type(e).__name__}:** {e}")
                    except Exception as e:
                        st.error(f"**Error:** {e}")

    # ── Raw CSV ───────────────────────────────────────────────────────────────
    elif input_mode == "📝  Raw CSV":
        st.markdown("Paste **raw CSV**.")
        st.caption("First column = time index, rest = series.")
        csv_input = st.text_area("CSV here", height=200,
                                 placeholder="Year,A,B\n2000,100,80\n2001,110,85\n…",
                                 key="csv_raw_input")
        csv_title = st.text_input("Chart title (optional)", key="csv_title_input",
                                  placeholder="e.g. GDP Race 2000–2024")
        if st.button("📥 Load CSV", use_container_width=True):
            if not csv_input.strip():
                st.warning("Paste CSV data first.")
            else:
                try:
                    df, title = parse_csv_text(csv_input, csv_title)
                    _store_df(df, title, title)
                    st.success(f"✅ Loaded **{title}**")
                except (DataIndexError, ResamplingFrequencyMismatch) as e:
                    st.error(f"**{type(e).__name__}:** {e}")
                except Exception as e:
                    st.error(f"**Error:** {e}")

    # ── Data preview ──────────────────────────────────────────────────────────
    if st.session_state["pending_df"] is not None:
        st.divider()
        pdf = st.session_state["pending_df"]
        with st.expander(f"📊 Data: {st.session_state['pending_title']}", expanded=False):
            st.dataframe(pdf.head(10), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state["pending_df"] is None:
    st.markdown("## 🎬 Topic-to-Reel")
    st.markdown(f"*{BRAND}* — Generate a **1080×1920 animated Reel** from any topic.")
    st.info("👈 Pick a data source from the sidebar to get started.")
else:
    pdf   = st.session_state["pending_df"]
    cols  = list(pdf.columns)
    n_col = len(cols)

    # ── Top header ────────────────────────────────────────────────────────────
    st.markdown(f"## {st.session_state.get('custom_title') or st.session_state['pending_title']}")

    # ── Two-column layout: customise | preview ────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        # ── Chart type ───────────────────────────────────────────────────────
        st.markdown("#### ⚙️ Chart Settings")
        chart_type = st.radio(
            "Chart type",
            ["📈  Line Chart Race", "📊  Bar Chart Race"],
            horizontal=True,
        )
        is_bar = "Bar" in chart_type

        if is_bar:
            bar_duration = st.slider("Duration (seconds)", 5, 60, 20, 5)
            st.caption(f"⏱ {bar_duration}s · {len(pdf)} periods")
        else:
            steps = st.slider("Frames per period (higher = slower)", 8, 60, 28, 4)
            n_p   = len(pdf)
            est   = (n_p - 1) * steps / FPS
            st.caption(f"⏱ ≈ {est:.0f}s total · {n_p} periods")

        # ── Re-render preview when chart type changes ─────────────────────────
        if (st.session_state.get("pending_preview_bytes")
                and is_bar != st.session_state.get("_preview_is_bar", False)):
            try:
                st.session_state["pending_preview_bytes"] = render_preview_frame(
                    pdf,
                    st.session_state.get("pending_labels") or [str(v) for v in pdf.index],
                    st.session_state.get("custom_title") or st.session_state["pending_title"],
                    {
                        "prefix":      st.session_state.get("custom_unit_prefix", ""),
                        "suffix":      st.session_state.get("custom_unit_suffix", ""),
                        "description": st.session_state.get("custom_subtitle", ""),
                        "is_pct":      st.session_state["pending_units"].get("is_pct", False),
                    },
                    is_bar=is_bar,
                    colors=st.session_state.get("custom_colors") or DEFAULT_COLORS,
                    x_label=st.session_state.get("custom_x_label", ""),
                    y_label=st.session_state.get("custom_y_label", ""),
                )
                st.session_state["_preview_is_bar"] = is_bar
            except Exception:
                pass

        st.divider()

        # ── Metadata editor ───────────────────────────────────────────────────
        st.markdown("#### ✏️ Customise Labels")

        st.session_state["custom_title"] = st.text_input(
            "Chart title",
            value=st.session_state.get("custom_title") or st.session_state["pending_title"],
        )
        st.session_state["custom_subtitle"] = st.text_input(
            "Subtitle / unit description",
            value=st.session_state.get("custom_subtitle", ""),
            placeholder="e.g. GDP in USD Billions",
        )

        c1, c2 = st.columns(2)
        with c1:
            st.session_state["custom_x_label"] = st.text_input(
                "X-axis label",
                value=st.session_state.get("custom_x_label", ""),
                placeholder="e.g. Year",
            )
            st.session_state["custom_unit_prefix"] = st.text_input(
                "Value prefix",
                value=st.session_state.get("custom_unit_prefix", ""),
                placeholder="e.g. $",
            )
        with c2:
            st.session_state["custom_y_label"] = st.text_input(
                "Y-axis label",
                value=st.session_state.get("custom_y_label", ""),
                placeholder="e.g. Value (Billions)",
            )
            st.session_state["custom_unit_suffix"] = st.text_input(
                "Value suffix",
                value=st.session_state.get("custom_unit_suffix", ""),
                placeholder="e.g. B or %",
            )

        st.divider()

        # ── Series names + colors ─────────────────────────────────────────────
        st.markdown("#### 🎨 Series Names & Colors")
        n_show = min(n_col, 8)
        series_cols_ui = st.columns(2)
        custom_series = st.session_state.get("custom_series_names", {col: col for col in cols})
        custom_colors = st.session_state.get("custom_colors") or DEFAULT_COLORS[:n_show]

        new_series_names: dict = {}
        new_colors: list = list(custom_colors)

        for i, col in enumerate(cols[:n_show]):
            with series_cols_ui[i % 2]:
                new_name = st.text_input(
                    f"Series {i+1}",
                    value=custom_series.get(col, col),
                    key=f"sname_{i}",
                    label_visibility="collapsed",
                )
                new_colors[i] = st.color_picker(
                    f"Color {i+1}",
                    value=custom_colors[i] if i < len(custom_colors) else DEFAULT_COLORS[i % 8],
                    key=f"cpick_{i}",
                    label_visibility="collapsed",
                )
                new_series_names[col] = new_name

        st.session_state["custom_series_names"] = new_series_names
        st.session_state["custom_colors"]        = new_colors

        # Custom icons (line chart only)
        if not is_bar:
            with st.expander("🖼️ Custom icons (optional)", expanded=False):
                ic_cols = st.columns(min(n_show, 4))
                for i, col in enumerate(cols[:n_show]):
                    with ic_cols[i % 4]:
                        up = st.file_uploader(
                            new_series_names.get(col, col)[:8],
                            type=["png","jpg","jpeg","webp"],
                            key=f"icon_{col}",
                        )
                        if up is not None:
                            st.session_state["custom_icons"][col] = up.read()
                        if col in st.session_state["custom_icons"]:
                            try:
                                st.image(Image.open(io.BytesIO(
                                    st.session_state["custom_icons"][col])).resize((36,36)),
                                    width=36)
                            except Exception:
                                pass
                            if st.button("✕", key=f"rm_{col}"):
                                del st.session_state["custom_icons"][col]
                                st.rerun()

    with right:
        st.markdown("#### 🖼️ Chart Preview")
        if st.session_state.get("pending_preview_bytes"):
            # Build effective units from custom fields
            eff_units = {
                "prefix":      st.session_state.get("custom_unit_prefix", "")
                               or st.session_state["pending_units"].get("prefix", ""),
                "suffix":      st.session_state.get("custom_unit_suffix", "")
                               or st.session_state["pending_units"].get("suffix", ""),
                "description": st.session_state.get("custom_subtitle", "")
                               or st.session_state["pending_units"].get("description", ""),
                "is_pct":      st.session_state["pending_units"].get("is_pct", False),
            }

            if st.button("🔄 Refresh preview", use_container_width=True):
                # Rename columns to custom names before preview
                df_preview = pdf.rename(columns=st.session_state["custom_series_names"])
                try:
                    preview = render_preview_frame(
                        df_preview,
                        st.session_state.get("pending_labels") or [str(v) for v in pdf.index],
                        st.session_state.get("custom_title") or st.session_state["pending_title"],
                        eff_units,
                        is_bar=is_bar,
                        colors=st.session_state.get("custom_colors") or DEFAULT_COLORS,
                        x_label=st.session_state.get("custom_x_label", ""),
                        y_label=st.session_state.get("custom_y_label", ""),
                    )
                    st.session_state["pending_preview_bytes"] = preview
                    st.session_state["_preview_is_bar"] = is_bar
                except Exception as e:
                    st.warning(f"Preview failed: {e}")

            col_l, col_c, col_r = st.columns([0.5, 9, 0.5])
            with col_c:
                st.image(
                    st.session_state["pending_preview_bytes"],
                    caption="Final frame preview",
                    use_container_width=True,
                )
        else:
            st.info("Preview will appear here after loading data.")

    # ── Generate button ───────────────────────────────────────────────────────
    st.divider()
    generate = st.button(
        "🎬  Generate Reel",
        type="primary",
        use_container_width=True,
        disabled=(st.session_state["pending_df"] is None),
    )

    if generate:
        df          = st.session_state["pending_df"]
        idx_labels  = st.session_state.get("pending_labels") or [str(v) for v in df.index]
        topic_str   = st.session_state["pending_topic"] or st.session_state["pending_title"]

        # Apply custom series names
        snames = st.session_state.get("custom_series_names", {})
        df_use = df.rename(columns=snames)
        chart_title = st.session_state.get("custom_title") or st.session_state["pending_title"]
        n_lines = min(len(df_use.columns), 8)
        df_use  = df_use.iloc[:, :n_lines]
        eff_colors = (st.session_state.get("custom_colors") or DEFAULT_COLORS)[:n_lines]

        eff_units = {
            "prefix":      st.session_state.get("custom_unit_prefix", "")
                           or st.session_state["pending_units"].get("prefix", ""),
            "suffix":      st.session_state.get("custom_unit_suffix", "")
                           or st.session_state["pending_units"].get("suffix", ""),
            "description": st.session_state.get("custom_subtitle", "")
                           or st.session_state["pending_units"].get("description", ""),
            "is_pct":      st.session_state["pending_units"].get("is_pct", False),
        }

        x_lbl = st.session_state.get("custom_x_label", "")
        y_lbl = st.session_state.get("custom_y_label", "")

        progress = st.progress(0, text="Starting…")
        status   = st.empty()

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                raw_mp4   = os.path.join(tmp_dir, "race.mp4")
                final_mp4 = os.path.join(tmp_dir, "reel.mp4")

                progress.progress(10, text="Fetching icons…")
                status.info("🏳  Fetching category icons…")
                icon_arrays = get_icons(
                    list(df_use.columns), eff_colors, size=48,
                    custom_icons=st.session_state.get("custom_icons", {}),
                )

                progress.progress(22, text="Rendering animation…")
                if is_bar:
                    status.info(f"📊  Rendering bar race — {len(df_use)} periods")
                    create_bar_race_video(
                        df_use, idx_labels, chart_title,
                        icon_arrays, eff_units, bar_duration, raw_mp4,
                        colors=eff_colors, x_label=x_lbl, y_label=y_lbl,
                    )
                else:
                    n_frames = (len(df_use)-1) * steps + 1
                    status.info(f"📈  Rendering line race — {n_frames} frames")
                    create_line_race_video(
                        df_use, idx_labels, chart_title,
                        icon_arrays, eff_units, steps, raw_mp4,
                        colors=eff_colors, x_label=x_lbl, y_label=y_lbl,
                    )

                progress.progress(78, text="Adding music…")
                status.info("🎵  Adding background music…")
                post_produce(raw_mp4, final_mp4, tmp_dir)

                progress.progress(90, text="Generating caption…")
                status.info("✍️  Writing Instagram caption…")
                caption, hashtags = generate_caption(topic_str, chart_title, df_use, eff_units)

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

# ── Output ─────────────────────────────────────────────────────────────────────
if st.session_state["last_video"]:
    video_bytes = st.session_state["last_video"]
    st.divider()
    out_l, out_r = st.columns([1, 1], gap="large")
    with out_l:
        st.subheader("🎬 Your Reel")
        st.video(video_bytes)
        st.download_button(
            label="⬇️  Download MP4  (1080 × 1920)",
            data=video_bytes,
            file_name=f"reel_{int(time.time())}.mp4",
            mime="video/mp4",
            use_container_width=True,
        )
    with out_r:
        st.subheader("📝 Caption")
        st.caption("Copy & paste to Instagram / TikTok")
        hashtag_line = " ".join(f"#{h}" for h in st.session_state["last_hashtags"])
        st.code(f"{st.session_state['last_caption']}\n\n{hashtag_line}", language=None)

# ── History ─────────────────────────────────────────────────────────────────────
if st.session_state["history"]:
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
