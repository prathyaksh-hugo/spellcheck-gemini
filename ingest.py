# ingest.py
import os
import json
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

print("Starting knowledge base ingestion...")

# --- Configuration ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found.")
genai.configure(api_key=api_key)

KNOWLEDGE_BASE_DIR = "data/knowledge_bases"
DB_PATH = "db"
COLLECTION_NAME = "unified_knowledge_base"

# --- Database Setup ---
client = chromadb.PersistentClient(path=DB_PATH)
if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    client.delete_collection(name=COLLECTION_NAME)
    print(f"Deleted existing collection: {COLLECTION_NAME}")
collection = client.create_collection(name=COLLECTION_NAME)
print(f"Created new collection: {COLLECTION_NAME}")

# --- Data Processing and Ingestion ---
all_contents = []
all_metadatas = []
all_ids = []
doc_id_counter = 0

# Read all .json files from the knowledge base directory
for filename in os.listdir(KNOWLEDGE_BASE_DIR):
    if filename.endswith(".json"):
        source_id = filename.replace(".json", "")
        print(f"Processing source: {source_id}...")
        
        with open(os.path.join(KNOWLEDGE_BASE_DIR, filename), 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                all_contents.append(item["content"])
                # Add the source_id to the metadata
                all_metadatas.append({"source": source_id})
                all_ids.append(f"doc_{doc_id_counter}")
                doc_id_counter += 1

if not all_contents:
    print("No documents found to ingest.")
else:
    print(f"Found {len(all_contents)} total rules to process.")
    # Generate embeddings in a single batch
    print("Generating embeddings...")
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=all_contents,
        task_type="RETRIEVAL_DOCUMENT"
    )
    embeddings = result['embedding']
    print("Embeddings generated.")

    # Add to ChromaDB
    collection.add(
        embeddings=embeddings,
        documents=all_contents,
        metadatas=all_metadatas,
        ids=all_ids
    )
    print(f"✅ Successfully ingested {collection.count()} rules into the database.")