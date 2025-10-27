# src/main.py
import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------
CHROMA_DIR = os.path.join(os.getcwd(), "memory", "chroma_db")
MODEL_PATH = "./models/all-MiniLM-L6-v2"
MEMORY_JSON = "../data/sample_memories.json"

# --------------------------------------------------------
# INIT CHROMA CLIENT (new API)
# --------------------------------------------------------
def init_client():
    """
    Initialize a persistent Chroma client using the latest API.
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client

# --------------------------------------------------------
# BUILD MEMORY COLLECTION
# --------------------------------------------------------
def build_memory_collection(client, model, json_path=MEMORY_JSON):
    """
    Build or update a persistent Chroma memory collection using embeddings.
    """
    collection_name = "memory"

    # Get or create collection
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        collection = client.create_collection(name=collection_name)

    # Load sample memory data
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    texts = [it["text"] for it in items]
    ids = [it["id"] for it in items]

    # Encode embeddings
    print("🔄 Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # Fetch existing IDs to avoid duplicates
    existing_ids = set()
    if collection.count() > 0:
        try:
            all_docs = collection.get(include=[])
            existing_ids = set(all_docs["ids"])
        except Exception:
            pass

    # Add only new items
    new_entries = [
        (eid, emb, txt)
        for eid, emb, txt in zip(ids, embeddings, texts)
        if eid not in existing_ids
    ]

    if new_entries:
        print(f"➕ Adding {len(new_entries)} new items to memory collection...")
        collection.add(
            ids=[e[0] for e in new_entries],
            embeddings=[e[1] for e in new_entries],
            documents=[e[2] for e in new_entries],
        )
        print("✅ Memory updated successfully.")
    else:
        print("ℹ️ No new items to add — collection already up-to-date.")

    return collection

# --------------------------------------------------------
# QUERY MEMORY COLLECTION
# --------------------------------------------------------
def query_memory(collection, model, query_text, n_results=3):
    """
    Query the Chroma memory collection using semantic similarity.
    """
    query_emb = model.encode([query_text]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=n_results)

    print("\n🧠 Query:", query_text)
    print("🔍 Top Results:")
    for i, doc in enumerate(results["documents"][0]):
        print(f"{i+1}. {doc}")

    return results

# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Loading local embedding model...")
    model = SentenceTransformer(MODEL_PATH)
    client = init_client()
    collection = build_memory_collection(client, model)

    # Example query
    query_text = "automata theory"
    query_memory(collection, model, query_text)
