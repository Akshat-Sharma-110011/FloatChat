"""Build two Chroma collections: kb_schema and kb_summaries.
Run:
  python -m src.build_vectorstore
"""
import os, json, chromadb, pathlib
from chromadb.utils import embedding_functions
from .config import DATA_DIR

def main():
    client = chromadb.PersistentClient(path=str(pathlib.Path(DATA_DIR) / "chroma"))
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    kb_schema = client.get_or_create_collection(name="kb_schema", embedding_function=embedder)
    kb_summ = client.get_or_create_collection(name="kb_summaries", embedding_function=embedder)

    # Minimal schema doc
    schema_text = """
    Tables:
    - floats(platform_number TEXT PRIMARY KEY?, first_time TIMESTAMP, last_time TIMESTAMP, n_profiles INT)
    - profiles(profile_id INT PRIMARY KEY, platform_number TEXT, cycle_number INT, latitude DOUBLE, longitude DOUBLE, time TIMESTAMP)
    - observations(profile_id INT, level INT, pressure_dbar DOUBLE, temperature_c DOUBLE, salinity_psu DOUBLE)
    Relationships:
    observations.profile_id -> profiles.profile_id
    profiles.platform_number -> floats.platform_number
    Common tasks: filter by time range, lat/lon box, variable != NULL, aggregate or plot by depth/time.
    """
    kb_schema.upsert(
        ids=["schema_v1"],
        documents=[schema_text],
        metadatas=[{"type":"schema","version":"v1"}]
    )

    # Summaries
    summ_path = pathlib.Path(DATA_DIR) / "kb_summaries.jsonl"
    if summ_path.exists():
        ids, docs, metas = [], [], []
        with open(summ_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                rec = json.loads(line)
                ids.append(rec["id"])
                docs.append(rec["text"])
                metas.append({"type":"summary"})
        if ids:
            kb_summ.upsert(ids=ids, documents=docs, metadatas=metas)

    print("Vector stores ready at ./data/chroma")

if __name__ == "__main__":
    main()
