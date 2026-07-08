# GenAI Developer · Contract Flight Deck

[![deploy](https://github.com/duttsiddharth/genaisimulation/actions/workflows/deploy.yml/badge.svg)](https://github.com/duttsiddharth/genaisimulation/actions/workflows/deploy.yml)
[![ci](https://github.com/duttsiddharth/genaisimulation/actions/workflows/ci.yml/badge.svg)](https://github.com/duttsiddharth/genaisimulation/actions/workflows/ci.yml)
[![live demo](https://img.shields.io/badge/demo-live-22d3ee.svg)](https://duttsiddharth.github.io/genaisimulation/)

An interactive simulator to **practice, learn and deliver** on a GenAI Developer role. The retrieval engine is real — chunking, TF-IDF + BM25 hybrid scoring and ranking all run in the browser — so it's a working rig, not a mockup.

**▶ Live demo: https://duttsiddharth.github.io/genaisimulation/**

---

## What's inside

- **Flight Deck** — a readiness gauge that aggregates every module you complete, with the job description mapped to concrete capabilities.
- **RAG Lab** — paste any source, pick a chunking strategy (fixed / sentence / recursive), tune size + overlap, then run live **hybrid retrieval**: TF-IDF cosine fused with BM25, an adjustable α weighting, ranked chunks with per-signal scores, and a guarded prompt assembled from the results.
- **ETL Studio** — toggle clean / dedup / PII-redact / drop / embed stages and watch messy multi-source data become an embeddable knowledge base.
- **API Forge** — configure a route, model, streaming, RAG injection and auth to generate a production-shaped FastAPI scaffold.
- **Agent Loop** — step through a reason → act → observe cycle over a live trace.
- **MLOps Pipeline** — the CI/CD, monitoring, drift and lifecycle checklist as an operability tracker.
- **Practice Arena** — concept quiz with explanations plus coding katas (recursive chunking, reciprocal rank fusion, guarded prompts) with reveal-on-demand solutions.
- **Delivery Kit** — a 30/60/90 plan and a job-description-mapped interview drill.

## Run locally

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # production build to dist/
```

## Deploy to GitHub Pages

Deployment is automated by `.github/workflows/deploy.yml`. One-time setup:

1. Push to `main`.
2. Repo **Settings → Pages → Build and deployment → Source: GitHub Actions**.

Every push to `main` then builds and publishes to `https://duttsiddharth.github.io/genaisimulation/`. The `base` in `vite.config.js` is set to `/genaisimulation/` to match the repo name — if you rename the repo, update that value too.

## Notes

- Retrieval, scoring, chunking and the ETL and API tooling are entirely client-side and need no backend or keys.
- The RAG Lab's optional **Generate (live)** button calls a hosted model endpoint; on a static Pages deploy without a backend it degrades gracefully (retrieval still works, and you can copy the assembled prompt into your own stack). Wire it to your own API route to enable in-app generation.

## Stack

React 18 · Vite 5 · Tailwind CSS 3 · lucide-react. MIT licensed.
