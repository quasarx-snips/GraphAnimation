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

BG         = "#000000"
BRAND      = "worldstats.visualised"
FPS        = 30
STEPS      = 30
ICON_SIZE  = 56   # px for circular icon — smaller for neat label spacing


# ── Font loader ────────────────────────────────────────────────────────────────

def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in (FONT_BOLD, FONT_REG):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ── Icon helpers ───────────────────────────────────────────────────────────────

def _circle(pil_img: Image.Image, size: int = ICON_SIZE) -> np.ndarray:
    img  = pil_img.convert("RGBA").resize((size, size), Image.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
    img.putalpha(mask)
    return np.array(img)


def _initials_circle(color_hex: str, label: str, size: int = ICON_SIZE) -> np.ndarray:
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
    draw.ellipse((0, 0, size - 1, size - 1), fill=(r, g, b, 220))
    letter = label[0].upper()
    font   = _font(max(14, size // 2))
    bbox   = draw.textbbox((0, 0), letter, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size - tw) // 2, (size - th) // 2), letter,
              fill=(255, 255, 255, 255), font=font)
    return np.array(img)


def _flag_array(iso2: str, size: int = ICON_SIZE) -> np.ndarray | None:
    try:
        r = requests.get(f"https://flagcdn.com/48x36/{iso2.lower()}.png", timeout=6)
        if r.status_code == 200:
            return _circle(Image.open(io.BytesIO(r.content)), size)
    except Exception:
        pass
    return None


def identify_country_codes(cols: list[str]) -> dict[str, str | None]:
    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": (
            f"Given these chart category names: {cols}\n"
            "For each that is clearly a country return its ISO 3166-1 alpha-2 code (lowercase). "
            "For anything else return null.\n"
            'Return ONLY JSON like: {"USA": "us", "China": "cn", "Male": null}\n'
            "No markdown, no explanation."
        )}],
        max_completion_tokens=120,
    )
    try:
        raw = resp.choices[0].message.content.strip().strip("```json").strip("```").strip()
        return json.loads(raw)
    except Exception:
        return {c: None for c in cols}


def get_icons(cols: list[str], colors: list[str],
              size: int = ICON_SIZE) -> list[np.ndarray]:
    country_map = identify_country_codes(cols)
    icons = []
    for i, col in enumerate(cols):
        code = country_map.get(col)
        icon = _flag_array(code, size) if code else None
        if icon is None:
            icon = _initials_circle(colors[i], col, size)
        icons.append(icon)
    return icons


# ── Data + units extraction ────────────────────────────────────────────────────

def extract_data_from_llm(topic: str) -> tuple[pd.DataFrame, str]:
    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": "Return only clean CSV. No markdown. No explanations."},
            {"role": "user", "content": (
                f'Topic: "{topic}"\n\n'
                "Rules:\n"
                "- First column: numeric year (integer, consecutive, no gaps)\n"
                "- 2–6 category columns, short names ≤18 chars\n"
                "- Raw numeric values only — no commas inside numbers, no units, no symbols\n"
                "- Minimum 15 rows, ideally 20–30 rows\n"
                "- Realistic values that change over time\n"
                "- No missing values\n\n"
                "Return ONLY the CSV."
            )},
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


def detect_units(topic: str, df: pd.DataFrame) -> dict:
    """Ask LLM what unit/scale the values represent."""
    sample = {col: {"first": float(df[col].iloc[0]), "last": float(df[col].iloc[-1])}
              for col in list(df.columns)[:3]}
    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": (
            f'Topic: "{topic}"\nSample data values: {json.dumps(sample)}\n\n'
            "Determine the correct unit formatting for these values. Return ONLY JSON:\n"
            '{"prefix": "$", "scale": 1000000000, "scale_label": "B", '
            '"suffix": "", "description": "USD Billions"}\n\n'
            "prefix: currency symbol or empty string\n"
            "scale: divisor (1, 1e3, 1e6, 1e9, 1e12)\n"
            "scale_label: K / M / B / T or empty\n"
            "suffix: unit after number (e.g. '%', ' ppl') or empty\n"
            "description: short human-readable unit label"
        )}],
        max_completion_tokens=80,
    )
    try:
        raw = resp.choices[0].message.content.strip().strip("```json").strip("```").strip()
        u   = json.loads(raw)
        u.setdefault("prefix", "")
        u.setdefault("scale", 1)
        u.setdefault("scale_label", "")
        u.setdefault("suffix", "")
        u.setdefault("description", "")
        return u
    except Exception:
        return {"prefix": "", "scale": 1, "scale_label": "", "suffix": "", "description": ""}


def fmt(val: float, u: dict) -> str:
    """Auto-scale for readability; thresholds chosen so e.g. 800B → 0.80T on a T-scale chart."""
    p  = u.get("prefix", "")
    sf = u.get("suffix", "")
    av = abs(val)
    if av >= 5e11:   return f"{p}{val/1e12:.2f}T{sf}"   # ≥ 500 B → show as T
    if av >= 5e8:    return f"{p}{val/1e9:.2f}B{sf}"    # ≥ 500 M → show as B
    if av >= 5e5:    return f"{p}{val/1e6:.2f}M{sf}"    # ≥ 500 K → show as M
    if av >= 5e2:    return f"{p}{val/1e3:.1f}K{sf}"    # ≥ 500   → show as K
    if av >= 1:      return f"{p}{val:.1f}{sf}"
    return f"{p}{val:.2f}{sf}"


# ── Caption + hook title ───────────────────────────────────────────────────────

def generate_caption_and_hook(
    topic: str, chart_title: str, df: pd.DataFrame, units: dict
) -> tuple[str, list[str], str]:
    """Return (caption, hashtags, hook_title)."""

    # Build a compact data summary for the LLM
    cols = list(df.columns)
    years = list(df.index)
    summary_lines = [f"Period: {years[0]}–{years[-1]}, unit: {units['description']}"]
    for col in cols:
        start, end = df[col].iloc[0], df[col].iloc[-1]
        pct = ((end - start) / start * 100) if start != 0 else 0
        summary_lines.append(f"  {col}: {fmt(start, units)} → {fmt(end, units)} ({pct:+.1f}%)")
    summary = "\n".join(summary_lines)

    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": (
            f"You are a data scientist creating an Instagram reel about: '{topic}'\n"
            f"Chart title: '{chart_title}'\n"
            f"Data summary:\n{summary}\n\n"
            "Write three things separated by the markers below:\n\n"
            "CAPTION:\n"
            "3–5 sentences. Explain the key trend, a notable inflection point, "
            "growth rates, and a comparative insight — like a data scientist presenting "
            "to a curious public. Use the actual numbers. End with a one-line question "
            "that hooks the reader.\n\n"
            "HASHTAGS:\n"
            "Exactly 10 hashtags. MUST include: #fyp #viral #trending #dataviz "
            "#datavisualization — then 5 topic-specific ones. Space-separated, no numbering.\n\n"
            "HOOK:\n"
            "5–7 words. A punchy thumbnail headline (e.g. 'GDP Clash: Who Really Wins?'). "
            "Be dramatic. No emojis.\n\n"
            "Format exactly:\n"
            "CAPTION: <text>\n"
            "HASHTAGS: <tags>\n"
            "HOOK: <text>"
        )}],
        max_completion_tokens=500,
    )

    text      = resp.choices[0].message.content.strip()
    caption   = ""
    hashtags  = []
    hook      = chart_title   # fallback

    for line in text.splitlines():
        if line.startswith("CAPTION:"):
            caption = line[len("CAPTION:"):].strip()
        elif line.startswith("HASHTAGS:"):
            raw_tags = line[len("HASHTAGS:"):].strip().split()
            hashtags = [t.lstrip("#") for t in raw_tags if t.startswith("#")][:10]
        elif line.startswith("HOOK:"):
            hook = line[len("HOOK:"):].strip().strip('"\'')

    # Multi-line caption fallback
    if not caption and "CAPTION:" in text:
        parts   = text.split("CAPTION:")
        caption = parts[1].split("HASHTAGS:")[0].strip() if "HASHTAGS:" in parts[1] else parts[1].strip()

    if not hashtags:
        hashtags = ["fyp", "viral", "trending", "dataviz", "datavisualization"]

    return caption, hashtags, hook


# ── Animation ──────────────────────────────────────────────────────────────────

def create_line_race_video(
    df: pd.DataFrame,
    chart_title: str,
    icon_arrays: list[np.ndarray],
    units: dict,
    raw_path: str,
) -> str:

    n_periods = len(df)
    n_lines   = min(len(df.columns), len(LINE_COLORS))
    df        = df.iloc[:, :n_lines]
    cols      = list(df.columns)
    colors    = LINE_COLORS[:n_lines]

    total_frames = (n_periods - 1) * STEPS + 1
    x_raw        = np.arange(n_periods, dtype=float)
    x_interp     = np.linspace(0, n_periods - 1, total_frames)
    y_interp     = {c: np.interp(x_interp, x_raw, df[c].values) for c in cols}
    idx_labels   = [str(v) for v in df.index]

    FIG_W, FIG_H, DPI = 6.75, 12.0, 160
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG, dpi=DPI)
    gs  = fig.add_gridspec(
        2, 1, height_ratios=[0.11, 0.89],
        left=0.12, right=0.70, top=0.91, bottom=0.09, hspace=0.02,
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
                  color="white", fontsize=15, fontweight="bold",
                  ha="center", va="center")
    title_ax.text(0.5, 0.22, "↓  Read caption to know more  ↓",
                  transform=title_ax.transAxes,
                  color="#555555", fontsize=8, ha="center", va="center",
                  fontstyle="italic")

    # ── Chart styling ─────────────────────────────────────────────────────────
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#222222")
        ax.spines[sp].set_linewidth(0.8)
    ax.set_facecolor(BG)
    ax.yaxis.grid(True, color="#141414", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks([])

    # ── Axes limits ───────────────────────────────────────────────────────────
    # x normalized to [0, 1]; icons at x=1.0 (right axis boundary), clip_on=False
    ax.set_xlim(-0.05, 1.0)
    all_y  = np.concatenate([y_interp[c] for c in cols])
    y_min, y_max = all_y.min(), all_y.max()
    y_pad  = (y_max - y_min) * 0.10
    ax.set_ylim(y_min - y_pad, y_max + y_pad * 2.5)

    # Y-axis: auto-scaled unit labels (same logic as fmt())
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: fmt(v, units))
    )
    ax.tick_params(axis="y", labelcolor="#555555", labelsize=8)

    # ── Dynamic x-tick labels (text objects, updated each frame) ──────────────
    TICK_X = [0.0, 0.25, 0.5, 0.75, 1.0]
    tick_txts = [
        ax.text(tx, 0, "", color="#555555", fontsize=8, ha="center", va="top",
                transform=ax.get_xaxis_transform())
        for tx in TICK_X
    ]

    # ── Per-series artists ────────────────────────────────────────────────────
    main_lines, glow_lines, anno_boxes, val_labels = [], [], [], []
    ZOOM = 0.30   # smaller icon so legend name fits cleanly

    for i, col in enumerate(cols):
        c = colors[i]
        glow, = ax.plot([], [], color=c, linewidth=9, alpha=0.09,
                        solid_capstyle="round", zorder=2)
        main, = ax.plot([], [], color=c, linewidth=2.2,
                        solid_capstyle="round", zorder=3)

        imagebox = OffsetImage(icon_arrays[i], zoom=ZOOM)
        # xycoords='data', box_alignment centres icon on (x,y)
        ab = AnnotationBbox(
            imagebox, (0, 0),
            frameon=False, zorder=10,
            clip_on=False, pad=0,
            xycoords="data", box_alignment=(0.5, 0.5),
        )
        ab.set_visible(False)
        ax.add_artist(ab)

        # Value label: "Col: <value>"  — positioned to right of icon, outside axes
        lbl = ax.text(0, 0, "", color=c, fontsize=8.5, fontweight="bold",
                      va="center", ha="left", zorder=11, clip_on=False)

        glow_lines.append(glow)
        main_lines.append(main)
        anno_boxes.append(ab)
        val_labels.append(lbl)

    # ── Legend ────────────────────────────────────────────────────────────────
    ax.legend(
        handles=[plt.Line2D([0], [0], color=colors[i], linewidth=2, label=cols[i])
                 for i in range(n_lines)],
        loc="upper left", frameon=True, framealpha=0.18,
        facecolor="#0d0d0d", edgecolor="#222222",
        labelcolor="white", fontsize=8.5,
        handlelength=1.4, borderpad=0.6, labelspacing=0.4,
    )

    # ── Period counter ────────────────────────────────────────────────────────
    period_txt = ax.text(0.68, 0.05, "",
                         transform=ax.transAxes,
                         color="white", fontsize=32, fontweight="bold",
                         ha="right", va="bottom", alpha=0.75)

    # ── Branding ──────────────────────────────────────────────────────────────
    fig.text(0.5, 0.022, BRAND, ha="center", va="bottom",
             fontsize=7.5, color="#2a2a2a")

    # ── Update function ───────────────────────────────────────────────────────
    def update(frame: int):
        f         = min(frame, total_frames - 1)
        cur_x     = x_interp[f]
        p_idx     = int(np.clip(round(cur_x), 0, n_periods - 1))

        # Normalize so current position = 1.0 (compressing x-axis)
        if cur_x > 0:
            x_norm = x_interp[:f + 1] / cur_x
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
                # Label sits just past the axis boundary
                val_labels[i].set_position((icon_x + 0.04, yend))
                val_labels[i].set_text(f"{col}: {fmt(yend, units)}")
            else:
                anno_boxes[i].set_visible(False)
                val_labels[i].set_text("")

        # Update x-tick text
        for j, txt_obj in enumerate(tick_txts):
            mapped = min(int(TICK_X[j] * p_idx), n_periods - 1)
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


# ── Thumbnail (PIL countryball clash style) ────────────────────────────────────

def create_thumbnail_pil(
    cols: list[str],
    colors: list[str],
    icon_arrays: list[np.ndarray],   # ICON_SIZE circles
    chart_title: str,
    hook_title: str,
    units: dict,
) -> bytes:
    W, H = 1080, 1920
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    draw   = ImageDraw.Draw(canvas)

    n = len(cols)

    # ── Hook title ────────────────────────────────────────────────────────────
    # Split into at most 2 lines by finding the middle space
    words = hook_title.split()
    if len(words) > 3:
        mid   = len(words) // 2
        line1 = " ".join(words[:mid]).upper()
        line2 = " ".join(words[mid:]).upper()
    else:
        line1 = hook_title.upper()
        line2 = ""

    font_hook = _font(108)
    font_sub  = _font(58)
    font_name = _font(52)
    font_vs   = _font(64)
    font_info = _font(38)
    font_brd  = _font(34)

    # Top accent bar
    for i, col in enumerate(cols[:4]):
        seg_w = W // max(n, 1)
        c     = colors[i]
        r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
        draw.rectangle([i * seg_w, 0, (i + 1) * seg_w, 12], fill=(r, g, b))

    # Hook lines
    y_cursor = 80
    draw.text((W // 2, y_cursor), line1, fill="white",
              font=font_hook, anchor="mt", stroke_width=2, stroke_fill="#000000")
    y_cursor += 130
    if line2:
        draw.text((W // 2, y_cursor), line2, fill="#FFD700",
                  font=font_hook, anchor="mt", stroke_width=2, stroke_fill="#000000")
        y_cursor += 130

    # Divider
    y_cursor += 20
    draw.line([(80, y_cursor), (W - 80, y_cursor)], fill="#222222", width=2)
    y_cursor += 40

    # ── Circular icons ────────────────────────────────────────────────────────
    DISP = 340 if n <= 2 else (260 if n <= 4 else 200)

    if n == 1:
        positions = [(W // 2 - DISP // 2, y_cursor)]
    elif n == 2:
        gap = 70
        total = 2 * DISP + gap
        x0 = (W - total) // 2
        positions = [(x0, y_cursor), (x0 + DISP + gap, y_cursor)]
    elif n == 3:
        gap   = 40
        total = 3 * DISP + 2 * gap
        x0    = (W - total) // 2
        positions = [(x0 + i * (DISP + gap), y_cursor) for i in range(3)]
    else:
        # 2×2 grid (up to 4 shown)
        gap  = 40
        x0   = (W - 2 * DISP - gap) // 2
        y1   = y_cursor
        y2   = y_cursor + DISP + 60
        positions = [
            (x0, y1), (x0 + DISP + gap, y1),
            (x0, y2), (x0 + DISP + gap, y2),
        ]

    for i, col in enumerate(cols[:4]):
        if i >= len(positions):
            break
        px, py = positions[i]
        c = colors[i]
        r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)

        # Glow ring
        ring_pad = 16
        glow_img = Image.new("RGBA", (DISP + ring_pad * 2, DISP + ring_pad * 2), (0, 0, 0, 0))
        ImageDraw.Draw(glow_img).ellipse(
            (0, 0, DISP + ring_pad * 2 - 1, DISP + ring_pad * 2 - 1),
            fill=(r, g, b, 60)
        )
        canvas.paste(glow_img, (px - ring_pad, py - ring_pad), glow_img)

        # Resize icon to display size
        icon_pil = Image.fromarray(icon_arrays[i]).resize((DISP, DISP), Image.LANCZOS)
        canvas.paste(icon_pil, (px, py), icon_pil)

        # Category name below icon
        draw.text((px + DISP // 2, py + DISP + 18), col,
                  fill="white", font=font_name, anchor="mt")

    # "VS" badge(s) between pairs
    if n == 2:
        vs_x = W // 2
        vs_y = y_cursor + DISP // 2
        draw.ellipse(
            [(vs_x - 52, vs_y - 42), (vs_x + 52, vs_y + 42)],
            fill=(160, 20, 20)
        )
        draw.text((vs_x, vs_y), "VS", fill="white", font=font_vs, anchor="mm")

    # Advance cursor past icons + names
    if n <= 3:
        y_cursor += DISP + 90
    else:
        y_cursor += 2 * DISP + 150

    # ── Chart info strip ──────────────────────────────────────────────────────
    draw.line([(80, y_cursor), (W - 80, y_cursor)], fill="#222222", width=1)
    y_cursor += 36
    draw.text((W // 2, y_cursor), chart_title,
              fill="#AAAAAA", font=font_info, anchor="mt")
    y_cursor += 56
    if units.get("description"):
        draw.text((W // 2, y_cursor), f"Unit: {units['description']}",
                  fill="#555555", font=_font(30), anchor="mt")

    # ── Branding ──────────────────────────────────────────────────────────────
    draw.text((W // 2, H - 70), BRAND, fill="#333333", font=font_brd, anchor="mm")

    buf = io.BytesIO()
    canvas.convert("RGB").save(buf, format="PNG")
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

def _reddit_titles(sub: str, limit: int = 6) -> list[str]:
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
        raw.extend(_reddit_titles(sub, 5))
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
                    "(time-series comparisons). One per line, no bullets, no numbers."
                )},
                {"role": "user", "content": f"Trending:\n{block}\n\nGenerate 8 chart prompts."},
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
st.title("Topic-to-Reel")
st.caption("Enter a topic — get a 1080×1920 animated line chart Reel.")

# Trending suggestions
with st.spinner("Fetching today's trending topics…"):
    suggestions = get_trending_topics()

st.markdown("**Trending today — tap to use:**")

if "topic_value" not in st.session_state:
    st.session_state["topic_value"] = ""

btn_cols = st.columns(2)
for idx, sug in enumerate(suggestions):
    if btn_cols[idx % 2].button(sug, key=f"sug_{idx}", use_container_width=True):
        st.session_state["topic_value"] = sug
        st.rerun()

st.divider()

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

if st.button("Generate Reel", type="primary", use_container_width=True):
    if not topic.strip():
        st.warning("Please enter a topic first.")
        st.stop()

    progress = st.progress(0, text="Starting…")
    status   = st.empty()

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_mp4   = os.path.join(tmp_dir, "race.mp4")
            final_mp4 = os.path.join(tmp_dir, "reel.mp4")

            # 1 — Data
            progress.progress(5,  text="Researching data…")
            status.info("🔍  Researching and structuring data with AI…")
            df, chart_title = extract_data_from_llm(topic)

            # 2 — Units
            progress.progress(14, text="Detecting units…")
            status.info("📐  Detecting data units…")
            n_lines     = min(len(df.columns), len(LINE_COLORS))
            df          = df.iloc[:, :n_lines]
            cols_used   = list(df.columns)
            colors_used = LINE_COLORS[:n_lines]
            units       = detect_units(topic, df)

            # 3 — Icons
            progress.progress(20, text="Fetching icons…")
            status.info("🏳  Fetching category icons…")
            icon_arrays = get_icons(cols_used, colors_used)

            # 4 — Animation
            progress.progress(28, text="Rendering animation…")
            status.info(f"📈  Rendering compressing line race for: **{chart_title}**")
            create_line_race_video(df, chart_title, icon_arrays, units, raw_mp4)

            # 5 — Music
            progress.progress(78, text="Adding music…")
            status.info("🎬  Adding background music…")
            post_produce(raw_mp4, final_mp4, tmp_dir)

            # 6 — Caption + hook
            progress.progress(88, text="Generating caption…")
            status.info("✍️  Writing data-scientist caption and hashtags…")
            caption, hashtags, hook_title = generate_caption_and_hook(
                topic, chart_title, df, units
            )

            # 7 — Thumbnail
            progress.progress(95, text="Creating thumbnail…")
            status.info("🖼  Building thumbnail…")
            thumb_bytes = create_thumbnail_pil(
                cols_used, colors_used, icon_arrays, chart_title, hook_title, units
            )

            progress.progress(100, text="Done!")
            status.success("✅  Your Reel is ready!")

            with open(final_mp4, "rb") as f:
                video_bytes = f.read()

        # ── Output ────────────────────────────────────────────────────────────
        st.video(video_bytes)
        st.download_button(
            "⬇️  Download MP4 (1080×1920)", video_bytes,
            file_name=f"reel_{int(time.time())}.mp4", mime="video/mp4",
            use_container_width=True,
        )

        st.subheader("Thumbnail")
        st.image(thumb_bytes, use_container_width=True)
        st.download_button(
            "⬇️  Download Thumbnail (PNG)", thumb_bytes,
            file_name=f"thumb_{int(time.time())}.png", mime="image/png",
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
