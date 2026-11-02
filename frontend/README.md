# Cernovalis Frontend

Next.js 15 (App Router) + React 19 + TypeScript frontend for the Cernovalis fake news detection system. Dark, premium deep-space UI with a canvas-rendered AI data sphere.

## Setup

```bash
cd frontend
npm install
npm run dev        # http://localhost:3000
```

Start the backend (from the repo root) so the UI has data to talk to:

```bash
cd backend && source .venv/Scripts/activate
cd .. && .venv/Scripts/python -m uvicorn backend.app.main:app --reload --port 8000
```

(Equivalent single-liner: `source backend/.venv/Scripts/activate && uvicorn backend.app.main:app --reload --port 8000`)

## Environment

`frontend/.env`:

```env
NEXT_PUBLIC_API_URL=
```

- **Empty (default):** all requests go through a Next.js rewrite proxy (`/api/*` and `/health` → `http://localhost:8000`), so **no CORS changes are needed** in the backend.
- **Direct:** set `NEXT_PUBLIC_API_URL=http://localhost:8000` to call the backend directly. In that case add `http://localhost:3000` to `cors_origins` in `backend/app/config.py`.
- `BACKEND_ORIGIN` (optional, build-time) overrides the proxy destination for non-local deployments.
- The API base URL can also be changed at runtime in **Settings** (stored in localStorage).

## What's inside

| Route | Page |
|---|---|
| `/` | Landing with typewriter heading (Cernovalis / Quid est verum, quid est fictum?) and data sphere |
| `/detect` | Single article analysis, batch CSV upload, results + explanation panel |
| `/dashboard` | Stat cards, analysis trend, distribution donut (localStorage-backed) |
| `/model-comparison` | ISOT test-split metrics table, confusion matrix, dual-panel ROC, feature importance, training history |
| `/analytics` | Real vs fake over time, confidence distribution, error analysis, CSV/JSON export |
| `/history` | Searchable local history with detail modal |
| `/settings` | API base URL, default model, local data, about |

Data flows through a typed API layer (`lib/api.ts` + Zod schemas in `lib/schemas.ts`) mirroring the frozen backend contract: `POST /api/v1/analyze`, `POST /api/v1/batch`, `GET /health`. URL analysis is intentionally disabled in the UI (backend returns 501).

History is stored client-side in `localStorage` (key `cernovalis-history`). Backend history endpoints do not exist yet; wire them up in `lib/history.ts` when they ship.

Dashboard/model metrics come from `docs/model-card.md`; ROC curves and feature importances are illustrative renderings, and the numbers in the comparison table are real test-split results.
