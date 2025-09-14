# FloatChat Starter (SIH25040) — Agentic RAG over ARGO Data

This starter repo gives you an **end-to-end scaffold** to build a PoC:
- Ingest a small Indian Ocean subset from ARGO (NetCDF) → **Parquet**
- Load Parquet to **DuckDB** (fast local SQL) — swap to Postgres later if needed
- Build **two vector indexes** (schema docs + data summaries) with **Chroma**
- Minimal **Agentic RAG** (LLM + tools for SQL + plotting)
- **Streamlit** UI (chat + visualizations)

> PoC-first: DuckDB is used for speed in a hackathon. A Postgres schema is also provided if you want to deploy.
