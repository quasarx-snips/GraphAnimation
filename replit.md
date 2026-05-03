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
2. OpenAI LLM (via Replit AI Integrations) generates a wide-format CSV
3. `bar_chart_race` + Matplotlib renders a vertical 9:16 animation (1080×1920 @ 160 DPI)
4. MoviePy adds title overlay (Pillow-drawn) + background music
5. Final 1080×1920 MP4 available for preview and download

**Key dependencies:** streamlit, openai, pandas, bar_chart_race, matplotlib, moviepy, pillow

**Environment variables used:**
- `AI_INTEGRATIONS_OPENAI_BASE_URL` — Replit AI Integrations proxy URL
- `AI_INTEGRATIONS_OPENAI_API_KEY` — Replit AI Integrations key

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.
