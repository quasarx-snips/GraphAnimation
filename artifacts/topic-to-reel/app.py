"""
Topic-to-Reel — 1080×1920 animated line-chart race for Instagram Reels.
"""
from __future__ import annotations

import shutil, io, json, os, tempfile, time, calendar, traceback
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
_ff = shutil.which("ffmpeg")
if _ff:
    matplotlib.rcParams["animation.ffmpeg_path"] = _ff

import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import requests
import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import CubicSpline

# ── OpenAI client ──────────────────────────────────────────────────────────────
client = OpenAI(
    base_url=os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL"),
    api_key=os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY"),
)

# ── Constants ──────────────────────────────────────────────────────────────────
FONT_BOLD  = "/run/current-system/sw/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LINE_COLORS = [
    "#4FC3F7", "#FF7043", "#69F0AE", "#FFD740",
    "#E040FB", "#40C4FF", "#FF6D00", "#B9F6CA",
]
MUSIC_URLS = [
    "https://cdn.pixabay.com/download/audio/2021/09/23/audio_57bc8dcb4e.mp3",
    "https://cdn.pixabay.com/download/audio/2022/10/25/audio_d0bba6d4e2.mp3",
]
REDDIT_UA   = {"User-Agent": "TopicToReel/1.0"}
BRAND       = "worldstats.visualised"
BG          = "#000000"
FPS         = 30
ICON_PX     = 54        # target icon side-length in video pixels
# Gridspec margins (left/right/top/bottom in figure fraction)
AX_L, AX_R, AX_T, AX_B = 0.12, 0.66, 0.91, 0.09
ZOOM_SECS   = 3.0       # seconds for y-zoom intro
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

# ── Custom exception types ─────────────────────────────────────────────────────
class DataIndexError(ValueError):
    """Raised when the data has an invalid or inconsistent time index."""

class ResamplingFrequencyMismatch(ValueError):
    """Raised when temporal resampling produces an unexpected result."""

class RenderingError(RuntimeError):
    """Raised when the animation or export step fails."""

class IconFetchError(RuntimeError):
    """Raised when all icon sources fail for a given series."""

# ── Font helper ────────────────────────────────────────────────────────────────
def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_BOLD, size)
    except Exception:
        return ImageFont.load_default()

# ── Icon helpers ───────────────────────────────────────────────────────────────
def _to_circle(pil: Image.Image, size: int = 64) -> np.ndarray:
    img  = pil.convert("RGBA").resize((size, size), Image.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
    img.putalpha(mask)
    return np.array(img)

def _initials(color_hex: str, label: str, size: int = 64) -> np.ndarray:
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    r = int(color_hex[1:3], 16); g = int(color_hex[3:5], 16); b = int(color_hex[5:7], 16)
    draw.ellipse((0, 0, size - 1, size - 1), fill=(r, g, b, 215))
    letter = label[0].upper()
    font   = _font(max(14, size // 2))
    bb     = draw.textbbox((0, 0), letter, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    draw.text(((size - tw) // 2 - bb[0], (size - th) // 2 - bb[1]),
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
            'Example: {"India":"in","USA":"us","GDP":null}'
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

# ── Data parsing ───────────────────────────────────────────────────────────────
def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if not df.index.name or df.index.name == "":
        df = df.set_index(df.columns[0])
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.ffill().bfill().fillna(0)
    if len(df) < 2:
        raise DataIndexError("Data must have at least 2 rows (time periods).")
    return df

def extract_data_from_llm(topic: str) -> tuple[pd.DataFrame, str]:
    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": "Return ONLY clean CSV. No markdown."},
            {"role": "user",   "content": (
                f'Topic: "{topic}"\n'
                "Rules: first col = integer year (no gaps), 2–6 category cols (≤18 chars), "
                "raw numeric values only, 15–30 rows, realistic trends. Return ONLY CSV."
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
            {"role": "user",   "content": (
                "Convert this data into a clean CSV:\n"
                "- First column: time index (Year/Month/etc.)\n"
                "- Remaining: numeric series, raw values only\n\n"
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

# ── Temporal granularity ───────────────────────────────────────────────────────
def temporal_resample(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Auto-detect ΔT from index and resample:
      ΔT > 20 years  → annual (no change)
      1 < ΔT ≤ 20    → keep annual, just relabel
      ΔT ≤ 1         → weekly steps via spline
    Returns (new_df with integer 0-based index, idx_labels list).
    """
    idx = df.index.tolist()
    try:
        nums = [float(str(v).strip()) for v in idx]
    except (ValueError, TypeError):
        return df, [str(v) for v in idx]

    delta = nums[-1] - nums[0]

    if delta > 1:
        # Annual or multi-year — keep as-is
        return df, [str(v) for v in idx]

    elif 0 < delta <= 1:
        # Weekly interpolation (e.g. data spans ≤1 year)
        n_weeks = max(4, round(delta * 52) + 1)
        x_orig  = np.linspace(0, 1, len(df))
        x_fine  = np.linspace(0, 1, n_weeks)
        start_y = int(nums[0])
        labels  = [f"W{w + 1} {start_y}" for w in range(n_weeks)]
        new_data = {}
        for col in df.columns:
            cs = CubicSpline(x_orig, df[col].values.astype(float))
            new_data[col] = np.clip(cs(x_fine), df[col].min() * 0.8, df[col].max() * 1.2)
        new_df = pd.DataFrame(new_data, index=range(n_weeks))
        return new_df, labels

    else:
        raise ResamplingFrequencyMismatch(f"Index appears non-monotonic: delta={delta}")

# ── Units ──────────────────────────────────────────────────────────────────────
def detect_units(topic: str, df: pd.DataFrame) -> dict:
    sample = {c: {"first": float(df[c].iloc[0]), "last": float(df[c].iloc[-1])}
              for c in list(df.columns)[:3]}
    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": (
            f'Topic: "{topic}", sample values: {json.dumps(sample)}.\n'
            'Return ONLY JSON: {"prefix":"$","suffix":"","description":"USD Billions"}\n'
            "prefix=currency symbol or empty, suffix=%/ppl/etc. or empty, "
            "description=short human-readable unit label."
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
        s, e = df[col].iloc[0], df[col].iloc[-1]
        pct  = ((e - s) / s * 100) if s != 0 else 0
        lines.append(f"  {col}: {fmt(s, units)} → {fmt(e, units)} ({pct:+.1f}%)")

    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": (
            f"You are a data scientist creating an Instagram Reel about '{topic}'.\n"
            f"Chart: '{chart_title}'\nData:\n" + "\n".join(lines) + "\n\n"
            "Write a CAPTION (3-5 sentences): key trend, inflection point, growth rates, "
            "comparative insight — cite actual numbers. End with an engaging question.\n\n"
            "Then HASHTAGS: exactly 10. Must include #fyp #viral #trending #dataviz "
            "#datavisualization plus 5 topic-specific ones. Space-separated.\n\n"
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

# ── Collision avoidance ────────────────────────────────────────────────────────
def avoid_collisions(positions: list[float], min_gap: float,
                     iterations: int = 50) -> list[float]:
    """
    Force-directed y-nudging so labels don't overlap.
    Works in sorted order so items never swap — maintains relative ordering.
    """
    n = len(positions)
    if n <= 1:
        return list(positions)
    # Sort indices by original y value
    order = sorted(range(n), key=lambda i: positions[i])
    pos   = np.array([positions[order[i]] for i in range(n)], dtype=float)
    # Nudge in sorted order: pos[i] < pos[j] always (j > i)
    for _ in range(iterations):
        moved = False
        for i in range(n - 1):
            gap = pos[i + 1] - pos[i]
            if gap < min_gap:
                push      = (min_gap - gap) / 2.0 + 1e-10
                pos[i]   -= push
                pos[i+1] += push
                moved     = True
        if not moved:
            break
    # Map back to original indices
    result = [0.0] * n
    for sorted_i, orig_i in enumerate(order):
        result[orig_i] = float(pos[sorted_i])
    return result

# ── Animation ──────────────────────────────────────────────────────────────────
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

    # ── Catmull-Rom via CubicSpline (smooth between data points) ──────────────
    total_frames = (n_periods - 1) * steps_per_period + 1
    x_raw        = np.arange(n_periods, dtype=float)
    x_dense      = np.linspace(0, n_periods - 1, total_frames)

    # Vectorised precomputation — all series into a 2-D array
    y_interp: dict[str, np.ndarray] = {}
    for col in cols:
        raw_vals = df[col].values.astype(float)
        cs       = CubicSpline(x_raw, raw_vals)          # natural cubic spline
        interped = cs(x_dense)
        # Soft-clip so spline overshoots don't produce wild values
        y_interp[col] = np.clip(interped,
                                raw_vals.min() * 0.90,
                                raw_vals.max() * 1.10)

    # 2-D matrix shape (total_frames, n_lines) for fast column access
    Y_MAT = np.column_stack([y_interp[c] for c in cols])  # vectorised

    # ── Y-range from actual data ──────────────────────────────────────────────
    y_min_data = float(Y_MAT.min())
    y_max_data = float(Y_MAT.max())
    y_range    = max(y_max_data - y_min_data, 1.0)
    y_pad      = y_range * 0.10

    # Logarithmic scale if max/min ratio > 10× (and all positive)
    use_log = (y_min_data > 0) and ((y_max_data / y_min_data) > 10)

    if use_log:
        ylim_lo  = y_min_data * 0.5
        ylim_hi  = y_max_data * 2.0
        y_total  = ylim_hi - ylim_lo          # linear span used for icon sizing only
    else:
        ylim_lo  = y_min_data - y_pad
        ylim_hi  = y_max_data + y_pad * 3.0
        y_total  = ylim_hi - ylim_lo

    # ── Elastic X-axis boundaries ─────────────────────────────────────────────
    xlim_lo        = -0.35
    xlim_hi_max    = n_periods - 0.65
    ELASTIC_AHEAD  = 1.65   # periods ahead of current point kept visible

    # ── Zoom animation (Y only, cubic ease-out, first ZOOM_SECS s) ───────────
    zoom_frames = 0 if use_log else min(int(ZOOM_SECS * FPS), total_frames // 3)

    # ── Figure geometry ───────────────────────────────────────────────────────
    FIG_W, FIG_H, DPI = 6.75, 12.0, 160
    ax_w_px = (AX_R - AX_L) * FIG_W * DPI   # ≈ 626 px
    ax_h_px = (AX_T - AX_B) * FIG_H * DPI   # ≈ 1574 px

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
    # Centered chart title
    title_ax.text(0.50, 0.72, chart_title,
                  transform=title_ax.transAxes, color="#FFFFFF",
                  fontsize=15, fontweight="bold", ha="center", va="center")
    title_ax.text(0.50, 0.20, "↓  Read caption for full story  ↓",
                  transform=title_ax.transAxes, color="#666666",
                  fontsize=7.5, ha="center", va="center", fontstyle="italic")

    # ── Chart spine / grid ────────────────────────────────────────────────────
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#333333")
        ax.spines[sp].set_linewidth(0.8)

    if use_log:
        ax.set_yscale("log")
        ax.yaxis.grid(False)
    else:
        ax.yaxis.grid(True, color="#1A1A1A", linewidth=0.8, zorder=0)

    ax.set_axisbelow(True)
    ax.set_facecolor(BG)

    # ── Static axis configuration ─────────────────────────────────────────────
    # X ticks at every period; labels placed from actual idx_labels
    tick_step = max(1, n_periods // 6)
    tick_pos  = list(range(0, n_periods, tick_step))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([idx_labels[i] for i in tick_pos],
                       color="#CCCCCC", fontsize=8.5)
    ax.tick_params(axis="x", colors="#CCCCCC")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt(v, units)))
    ax.tick_params(axis="y", labelcolor="#CCCCCC", labelsize=8.5)

    # Y-axis description (elevated slightly)
    unit_desc = units.get("description", "")
    if unit_desc:
        ax.text(0.01, 0.998, f"Values in {unit_desc}",
                transform=ax.transAxes, color="#888888",
                fontsize=8, ha="left", va="top", fontstyle="italic")

    # Initial limits — elastic x starts at first 2 periods
    ax.set_xlim(xlim_lo, min(xlim_lo + ELASTIC_AHEAD + 1.0, xlim_hi_max))
    ax.set_ylim(ylim_lo, ylim_hi)

    # ── Legend (offset down for white-space) ─────────────────────────────────
    ax.legend(
        handles=[plt.Line2D([0], [0], color=colors[i], linewidth=2.2, label=cols[i])
                 for i in range(n_lines)],
        loc="upper left",
        bbox_to_anchor=(0.01, 0.84),    # <-- pushed down from 1.0
        frameon=True, framealpha=0.20,
        facecolor="#0A0A0A", edgecolor="#333333",
        labelcolor="#FFFFFF", fontsize=8.5,
        handlelength=1.4, borderpad=0.6, labelspacing=0.4,
    )

    # ── Period (year) counter — top-left, aligned with legend text ───────────
    period_txt = ax.text(
        0.01, 0.97, "",
        transform=ax.transAxes,
        color="#FFFFFF", fontsize=26, fontweight="bold",
        ha="left", va="top",
    )

    # ── Per-series artists ────────────────────────────────────────────────────
    outer_glow, inner_glow, main_lines, val_labels, icon_ims = [], [], [], [], []

    for i, col in enumerate(cols):
        c = colors[i]
        og, = ax.plot([], [], color=c, linewidth=18, alpha=0.07,
                      solid_capstyle="round", zorder=2)
        ig, = ax.plot([], [], color=c, linewidth=8,  alpha=0.18,
                      solid_capstyle="round", zorder=3)
        ml, = ax.plot([], [], color=c, linewidth=2.4,
                      solid_capstyle="round", zorder=4)
        # Icon via imshow — starts off-screen, moves along line tip
        im = ax.imshow(icon_arrays[i],
                       extent=[-99, -98, ylim_lo, ylim_lo + 1],
                       zorder=10, clip_on=False, aspect="auto",
                       interpolation="bilinear")
        lbl = ax.text(0, 0, "", color=c,
                      fontsize=8.5, fontweight="bold",
                      va="center", ha="left", zorder=12, clip_on=False)
        outer_glow.append(og); inner_glow.append(ig); main_lines.append(ml)
        icon_ims.append(im);   val_labels.append(lbl)

    # ── Branding ──────────────────────────────────────────────────────────────
    fig.text(0.5, 0.018, BRAND, ha="center", va="bottom",
             fontsize=7.5, color="#2E2E2E")

    # ── Update function ───────────────────────────────────────────────────────
    def update(frame: int):
        f          = min(frame, total_frames - 1)
        current_x  = float(x_dense[f])
        x_now_arr  = x_dense[:f + 1]
        p_idx      = int(np.clip(round(current_x), 0, n_periods - 1))

        # ── 1. Elastic X-axis: grows from initial 2-point window ──────────────
        xlim_right = min(current_x + ELASTIC_AHEAD, xlim_hi_max)
        ax.set_xlim(xlim_lo, xlim_right)
        cur_x_range = xlim_right - xlim_lo

        # ── 2. Y-axis zoom intro (cubic ease-out, 2x → 1x over zoom_frames) ──
        if not use_log and zoom_frames > 0 and f < zoom_frames:
            t_raw   = f / zoom_frames
            t_ease  = 1.0 - (1.0 - t_raw) ** 3      # cubic ease-out
            zoom    = 2.0 - t_ease                    # 2.0 → 1.0
            center  = float(Y_MAT[f].mean())
            half_h  = y_total / (2.0 * zoom)
            new_ylo = max(ylim_lo, center - half_h)
            new_yhi = min(ylim_hi, center + half_h)
            ax.set_ylim(new_ylo, new_yhi)
            cur_y_range = new_yhi - new_ylo
        else:
            if not use_log:
                ax.set_ylim(ylim_lo, ylim_hi)
            cur_y_range = y_total

        # ── 3. Dynamic icon size (recomputed from live axis limits) ───────────
        dx = ICON_PX / ax_w_px * cur_x_range
        dy = ICON_PX / ax_h_px * cur_y_range

        # ── 4. Gather raw label y-positions for collision detection ───────────
        raw_ypos: list[float] = []
        for i in range(n_lines):
            raw_ypos.append(float(Y_MAT[f, i]))

        nudged = avoid_collisions(raw_ypos, dy * 1.15)
        cur_ylim = ax.get_ylim()

        # ── 5. Per-series update ──────────────────────────────────────────────
        for i, col in enumerate(cols):
            y_now_arr = y_interp[col][:f + 1]
            outer_glow[i].set_data(x_now_arr, y_now_arr)
            inner_glow[i].set_data(x_now_arr, y_now_arr)
            main_lines[i].set_data(x_now_arr, y_now_arr)

            if len(x_now_arr) > 0:
                xend = float(x_now_arr[-1])
                yend = float(y_now_arr[-1])

                # Move icon along line tip
                icon_ims[i].set_extent([
                    xend - dx / 2, xend + dx / 2,
                    yend - dy / 2, yend + dy / 2,
                ])

                # Collision-nudged label, clamped inside ylim
                ny = float(np.clip(nudged[i],
                                   cur_ylim[0] + dy * 0.6,
                                   cur_ylim[1] - dy * 0.6))
                val_labels[i].set_position(
                    (xend + dx / 2 + cur_x_range * 0.018, ny))
                val_labels[i].set_text(
                    f"{col}: {fmt(float(Y_MAT[f, i]), units)}")
            else:
                icon_ims[i].set_extent([-99, -98, ylim_lo, ylim_lo + 1])
                val_labels[i].set_text("")

        # ── 6. Period counter (top-left) ──────────────────────────────────────
        period_txt.set_text(idx_labels[p_idx])

    # ── Render ────────────────────────────────────────────────────────────────
    try:
        ani = mpl_animation.FuncAnimation(
            fig, update, frames=total_frames,
            interval=1000 / FPS, blit=False,
        )
        writer = mpl_animation.FFMpegWriter(
            fps=FPS, codec="libx264",
            extra_args=[
                "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "17",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # ensure even dimensions
            ],
        )
        ani.save(raw_path, writer=writer, dpi=DPI,
                 savefig_kwargs={"facecolor": BG})
    except Exception as exc:
        raise RenderingError(f"Animation export failed: {exc}") from exc
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
                n = int(np.ceil(duration / audio.duration))
                audio = concatenate_audioclips([audio] * n)
            audio = audio.subclipped(0, duration).with_volume_scaled(0.30)
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
    for sub in ["dataisbeautiful", "worldnews", "science", "technology"]:
        raw.extend(_reddit(sub))
        if len(raw) >= 20:
            break
    if not raw:
        return FALLBACK_TOPICS
    block = "\n".join(f"- {t}" for t in raw[:24])
    try:
        resp = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content":
                    "Turn trending headlines into 8 animated line-chart prompts "
                    "(time-series comparisons). One per line, no bullets, no numbering."},
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
st.caption("Generate a 1080×1920 animated Reel from a topic, file, or your own data.")

# ── Session state init ────────────────────────────────────────────────────────
_defaults: dict = {
    "topic_value":   "",
    "pending_df":    None,
    "pending_title": "",
    "pending_topic": "",
    "pending_labels": None,     # idx_labels from temporal_resample
    "last_video":    None,
    "last_caption":  "",
    "last_hashtags": [],
    "history":       [],        # list of generation records
    "grid_df": pd.DataFrame({
        "Year":     [2020, 2021, 2022, 2023, 2024],
        "Series A": [100.0, 120.0, 145.0, 175.0, 210.0],
        "Series B": [80.0,  92.0, 108.0, 127.0, 150.0],
    }),
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Data input tabs ───────────────────────────────────────────────────────────
tab_ai, tab_upload, tab_paste, tab_grid = st.tabs([
    "🤖  AI Topic", "📁  Upload File", "📋  Paste Data", "📊  Grid Editor",
])

def _store_df(df: pd.DataFrame, title: str, topic: str) -> None:
    """Store loaded dataframe + run temporal resampling."""
    try:
        df_rs, labels = temporal_resample(df)
    except ResamplingFrequencyMismatch:
        df_rs, labels = df, [str(v) for v in df.index]
    st.session_state.update(
        pending_df=df_rs, pending_title=title,
        pending_topic=topic, pending_labels=labels,
    )

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

    topic_in = st.text_area("Or type your own topic", key="topic_value",
                             placeholder="e.g. US vs China GDP 1980–2024", height=80)
    if st.button("Load AI Data", key="load_ai", type="primary", use_container_width=True):
        if not topic_in.strip():
            st.warning("Enter a topic first.")
        else:
            with st.spinner("Researching data…"):
                try:
                    df, title = extract_data_from_llm(topic_in)
                    _store_df(df, title, topic_in)
                    st.success(f"Loaded: **{title}**  "
                               f"({len(st.session_state['pending_df'])} rows × "
                               f"{len(st.session_state['pending_df'].columns)} series)")
                except DataIndexError as e:
                    st.error(f"**DataIndexError:** {e}")
                except Exception as e:
                    st.error(f"**Error ({type(e).__name__}):** {e}")

# ── Tab 2: Upload ─────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown("Upload a **CSV** or **XLSX** file. First column = time index, rest = series.")
    uploaded = st.file_uploader("Choose file", type=["csv", "xlsx", "xls", "txt"],
                                key="file_upload")
    if uploaded:
        try:
            df, title = parse_uploaded_file(uploaded)
            _store_df(df, title, title)
            st.success(f"Loaded: **{title}**  "
                       f"({len(st.session_state['pending_df'])} rows × "
                       f"{len(st.session_state['pending_df'].columns)} series)")
            st.dataframe(st.session_state["pending_df"].head(8), use_container_width=True)
        except (DataIndexError, ResamplingFrequencyMismatch) as e:
            st.error(f"**{type(e).__name__}:** {e}")
        except Exception as e:
            st.error(f"**Error ({type(e).__name__}):** {e}")

# ── Tab 3: Paste ──────────────────────────────────────────────────────────────
with tab_paste:
    st.markdown("Paste **any data** — CSV, Wikipedia table, rough notes. AI structures it.")
    raw_paste = st.text_area("Paste data here", height=220,
                             placeholder="Year,India,USA\n2000,477,10300\n2005,820,13000\n…")
    if st.button("Parse & Load", key="parse_paste", use_container_width=True):
        if not raw_paste.strip():
            st.warning("Paste some data first.")
        else:
            with st.spinner("Parsing with AI…"):
                try:
                    df, title = parse_pasted_data(raw_paste)
                    _store_df(df, title, title)
                    st.success(f"Loaded: **{title}**  "
                               f"({len(st.session_state['pending_df'])} rows × "
                               f"{len(st.session_state['pending_df'].columns)} series)")
                    st.dataframe(st.session_state["pending_df"].head(8), use_container_width=True)
                except (DataIndexError, ResamplingFrequencyMismatch) as e:
                    st.error(f"**{type(e).__name__}:** {e}")
                except Exception as e:
                    st.error(f"**Error ({type(e).__name__}):** {e}")

# ── Tab 4: Grid editor ────────────────────────────────────────────────────────
with tab_grid:
    st.markdown("Edit the table directly. **First column = time index**, rest = series.")
    with st.expander("➕  Add a new column"):
        c1, c2 = st.columns([3, 1])
        new_col = c1.text_input("Name", key="new_col_name",
                                label_visibility="collapsed", placeholder="New series…")
        if c2.button("Add", key="add_col") and new_col.strip():
            st.session_state["grid_df"][new_col.strip()] = 0.0
            st.rerun()

    edited = st.data_editor(st.session_state["grid_df"], num_rows="dynamic",
                            use_container_width=True, key="data_editor_widget")
    chart_title_input = st.text_input("Chart title (optional)",
                                      placeholder="e.g. Global GDP Race")
    if st.button("Use Grid Data", key="use_grid", use_container_width=True):
        try:
            df    = edited.copy().set_index(edited.columns[0])
            df    = df.apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0)
            title = chart_title_input.strip() or "Custom Data Reel"
            _store_df(df, title, title)
            st.session_state["grid_df"] = edited
            st.success(f"Grid loaded: **{title}**  "
                       f"({len(st.session_state['pending_df'])} rows × "
                       f"{len(st.session_state['pending_df'].columns)} series)")
        except (DataIndexError, ResamplingFrequencyMismatch) as e:
            st.error(f"**{type(e).__name__}:** {e}")
        except Exception as e:
            st.error(f"**Error ({type(e).__name__}):** {e}")

# ── Data preview ──────────────────────────────────────────────────────────────
if st.session_state["pending_df"] is not None:
    pdf = st.session_state["pending_df"]
    st.divider()
    with st.expander(f"📊 Data preview: **{st.session_state['pending_title']}** "
                     f"— {len(pdf)} rows × {len(pdf.columns)} series",
                     expanded=False):
        st.dataframe(pdf, use_container_width=True)

# ── Settings ──────────────────────────────────────────────────────────────────
st.divider()
with st.expander("⚙️  Settings", expanded=True):
    steps = st.slider(
        "Frames per time period  (higher = slower, smoother)",
        min_value=8, max_value=60, value=28, step=4,
    )
    n_p = len(st.session_state["pending_df"]) if st.session_state["pending_df"] is not None else 25
    est = (n_p - 1) * steps / FPS
    st.caption(
        f"⏱  Each period = **{steps/FPS:.1f}s** → "
        f"estimated length **{est:.0f}s** ({est/60:.1f} min)  |  "
        f"{n_p} time periods · smooth Catmull-Rom spline"
        + ("  ·  📈 log scale" if (
            st.session_state["pending_df"] is not None
            and (df_chk := st.session_state["pending_df"]) is not None
            and df_chk.values.min() > 0
            and df_chk.values.max() / df_chk.values.min() > 10
        ) else "")
    )

# ── Generate button ───────────────────────────────────────────────────────────
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

            progress.progress(5,  text="Detecting units…")
            status.info("📐  Detecting data units…")
            n_lines     = min(len(df.columns), len(LINE_COLORS))
            df_use      = df.iloc[:, :n_lines]
            colors_used = LINE_COLORS[:n_lines]
            units       = detect_units(topic_str, df_use)

            progress.progress(14, text="Fetching icons…")
            status.info("🏳  Fetching category icons…")
            icon_arrays = get_icons(list(df_use.columns), colors_used)

            progress.progress(22, text="Rendering animation…")
            n_frames = (len(df_use) - 1) * steps + 1
            status.info(
                f"📈  Rendering **{chart_title}**  —  "
                f"{len(df_use)} periods × {n_lines} series  ·  "
                f"{n_frames} frames  ·  Catmull-Rom spline"
            )
            create_line_race_video(
                df_use, idx_labels, chart_title,
                icon_arrays, units, steps, raw_mp4,
            )

            progress.progress(78, text="Adding music…")
            status.info("🎬  Adding background music…")
            post_produce(raw_mp4, final_mp4, tmp_dir)

            progress.progress(91, text="Generating caption…")
            status.info("✍️  Writing data-scientist caption…")
            caption, hashtags = generate_caption(topic_str, chart_title, df_use, units)

            progress.progress(100, text="Done!")
            status.success("✅  Reel ready!")

            with open(final_mp4, "rb") as fh:
                video_bytes = fh.read()

        # ── Cache in session state (prevents re-generation on download click) ─
        st.session_state["last_video"]    = video_bytes
        st.session_state["last_caption"]  = caption
        st.session_state["last_hashtags"] = hashtags

        # ── Edit history (last 5) ─────────────────────────────────────────────
        st.session_state["history"].append({
            "ts":     datetime.now().strftime("%H:%M:%S"),
            "title":  chart_title,
            "topic":  topic_str,
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

# ── Persistent output (cached — not re-generated on download click) ───────────
if st.session_state["last_video"]:
    st.divider()
    st.subheader("Your Reel")
    st.video(st.session_state["last_video"])

    # Download button uses cached bytes → no re-run of video generation
    st.download_button(
        "⬇️  Download MP4  (1080 × 1920)",
        st.session_state["last_video"],
        file_name=f"reel_{int(time.time())}.mp4",
        mime="video/mp4",
        use_container_width=True,
    )

    st.subheader("Caption — copy & paste to Instagram")
    hashtag_line = " ".join(f"#{h}" for h in st.session_state["last_hashtags"])
    st.code(f"{st.session_state['last_caption']}\n\n{hashtag_line}", language=None)

# ── Generation history panel ──────────────────────────────────────────────────
if st.session_state["history"]:
    st.divider()
    with st.expander(f"🕘  Generation history  ({len(st.session_state['history'])} entries)",
                     expanded=False):
        for i, rec in enumerate(reversed(st.session_state["history"])):
            st.markdown(
                f"**{rec['ts']}** — {rec['title']}  "
                f"·  {rec['rows']} periods · {rec['series']} series  "
                f"·  {rec['bytes'] / 1e6:.1f} MB"
            )
