# Workspace

## Overview

pnpm workspace monorepo using TypeScript + a standalone Python Streamlit app.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)
- **Python version**: 3.11
- **Streamlit app**: `artifacts/topic-to-reel/app.py` — Topic-to-Reel generator

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally
- `streamlit run artifacts/topic-to-reel/app.py --server.port 5000` — run Topic-to-Reel app

## Topic-to-Reel App

Located at `artifacts/topic-to-reel/`.

**Workflow:** Topic-to-Reel (port 5000)

**Pipeline:**
1. User enters a topic or pastes data
2. OpenAI gpt-4.1 (via Replit AI Integrations) generates a wide-format CSV with CURRENT_YEAR=2026
3. Matplotlib renders a vertical 9:16 animation (1080×1920 @ 160 DPI) in one of 4 chart types:
   - **Line Race** — cubic-spline animated line chart with Twemoji/flag icons on dot tips
   - **Bar Race** — horizontal bar chart race with icon thumbnails
   - **Pie Race** — animated donut (>4 series) or full pie (≤4), wedge-cleared per frame
   - **Spider/Radar** — morphing polygon radar chart with normalised per-spoke values
4. All chart types use cubic ease-in/out animation (`t²(3-2t)` smoothstep)
5. MoviePy adds background music; caption generated via GPT
6. Final 1080×1920 MP4 for download; batch ZIP with per-topic chart type selection

**Icon system:**
- Country/entity flags: `flagcdn.com/w40/{cc}.png`
- Keyword emoji: Twemoji CDN `jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/`
- `EMOJI_MAP` with 80+ keywords; fallback to coloured initials
- Preview uses `_fast_icon()` (no LLM call); video uses full `get_icons()` pipeline

**Batch UI:**
- Up to 10 topics, per-topic chart type selectbox (Line/Bar/Pie/Radar)
- Separate sliders for line frames/period and bar/pie/radar duration
- Single ZIP download

**Key dependencies:** streamlit, openai, pandas, matplotlib, moviepy, pillow, scipy

**Environment variables used:**
- `AI_INTEGRATIONS_OPENAI_BASE_URL` — Replit AI Integrations proxy URL
- `AI_INTEGRATIONS_OPENAI_API_KEY` — Replit AI Integrations key

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.
