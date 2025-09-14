import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
from dotenv import load_dotenv; load_dotenv()
from src.agent import answer

st.set_page_config(page_title="FloatChat", layout="wide")
st.title("FloatChat — Conversational ARGO Data (PoC)")
query = st.text_input("Ask about ARGO data (e.g., 'show profiles near 10N, 70E in 2023')")

if st.button("Run") and query:
    with st.spinner("Thinking..."):
        resp = answer(query)
    st.subheader("SQL")
    st.code(resp["sql"], language="sql")
    st.write(f"Rows: {resp['rows']}")
    st.write(resp["data_preview"])
    if "map" in resp:
        st.subheader("Map")
        st.components.v1.html(open(resp["map"], "r", encoding="utf-8").read(), height=520, scrolling=False)

st.caption("PoC uses DuckDB + Parquet locally. Replace LLM stub with OpenAI/HF for full Agentic RAG.")




