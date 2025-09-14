import os, json, chromadb, duckdb, pandas as pd
from chromadb.utils import embedding_functions
from .config import DUCKDB_PATH, DATA_DIR
from .tools import sql_run, plot_profiles_map, plot_profile_overlay

# Placeholder LLM call (you will wire OpenAI or HF)
def llm_complete(prompt: str) -> str:
    # Minimal offline-safe stub that just echoes prompt guidance.
    return "SELECT profile_id, latitude, longitude, time, platform_number FROM profiles LIMIT 100"

def retrieve_docs(q: str, coll):
    res = coll.query(query_texts=[q], n_results=3)
    docs = res.get("documents", [[]])[0]
    return "\n\n".join(docs)

def plan_sql(nl_query: str) -> str:
    client = chromadb.PersistentClient(path=os.path.join(DATA_DIR, "chroma"))
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    schema = client.get_or_create_collection(name="kb_schema", embedding_function=embedder)
    kb = retrieve_docs(nl_query, schema)
    system = open("prompts/text_to_sql_system.md","r",encoding="utf-8").read()
    prompt = f"""{system}

Natural language question:
{nl_query}

Schema/docs:
{kb}

Write only the SQL:
"""
    sql = llm_complete(prompt)
    # Safety: ensure select + limit
    s = sql.strip().rstrip(';')
    if not s.lower().startswith("select"):
        s = "SELECT profile_id, latitude, longitude, time, platform_number FROM profiles LIMIT 100"
    if "limit" not in s.lower():
        s += " LIMIT 500"
    return s

def answer(nl_query: str) -> dict:
    sql = plan_sql(nl_query)
    df = sql_run(sql)
    payload = {"sql": sql, "rows": len(df), "data_preview": df.head(5).to_dict(orient="records")}
    # Simple routing for plots
    if {"latitude","longitude"}.issubset(df.columns):
        payload["map"] = plot_profiles_map(df)
    return payload


