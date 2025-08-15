# ingest.py
import json
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
import os

print("Starting Brand Guide ingestion...")

# --- 1. Configure API and Database ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found.")
genai.configure(api_key=api_key)

client = chromadb.PersistentClient(path="db")

# We create a new collection for the guide to keep things clean
collection_name = "brand_voice_guide"
if collection_name in [c.name for c in client.list_collections()]:
    client.delete_collection(name=collection_name)
    print(f"Deleted existing collection: {collection_name}")

collection = client.create_collection(name=collection_name)
print(f"Created new collection: {collection_name}")

# --- 2. Read and Process JSON data ---
with open('data/brand_guide_knowledge_base.json', mode='r', encoding='utf-8') as file:
    guide_data = json.load(file)
    
    contents = []
    metadatas = []
    ids = []

    for i, item in enumerate(guide_data):
        contents.append(item["content"])
        # Store the rule type in the metadata
        metadatas.append({"rule_type": item["rule_type"], "source": item["source"]})
        ids.append(f"rule_{i}")

    print(f"Found {len(contents)} rules to process.")

# --- 3. Generate Embeddings ---
print("Generating embeddings with Gemini...")
result = genai.embed_content(
    model="models/text-embedding-004",
    content=contents,
    task_type="RETRIEVAL_DOCUMENT" # Use RETRIEVAL_DOCUMENT for storing
)
embeddings = result['embedding']
print("Embeddings generated.")

# --- 4. Add to ChromaDB ---
collection.add(
    embeddings=embeddings,
    documents=contents,
    metadatas=metadatas,
    ids=ids
)

print(f"✅ Successfully ingested {collection.count()} rules into the database.")