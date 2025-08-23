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

if not os.path.isdir(KNOWLEDGE_BASE_DIR):
    print(f"Error: Knowledge base directory not found at '{KNOWLEDGE_BASE_DIR}'")
    print("Please ensure the directory exists and contains your .json knowledge files.")
    exit()

# --- Database Setup ---
client = chromadb.PersistentClient(path=DB_PATH)
if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    print(f"Collection '{COLLECTION_NAME}' already exists. Deleting it.")
    client.delete_collection(name=COLLECTION_NAME)
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
        # The source_id will be the filename without the extension
        # e.g., "spelling_and_terminology" or "grammar_and_style"
        source_id = filename.replace(".json", "")
        print(f"Processing source: {source_id}...")
        
        file_path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                # --- THIS IS THE KEY CHANGE ---
                # Construct a descriptive document from the structured JSON
                try:
                    guideline = item.get("guideline", "No guideline provided.")
                    incorrect = item.get("incorrect_example", "N/A")
                    correct = item.get("correct_example", "N/A")
                    explanation = item.get("explanation", "No explanation provided.")

                    # This creates a rich, searchable string for the database
                    content = (
                        f"Guideline: {guideline} "
                        f"Incorrect Example: '{incorrect}'. "
                        f"Correct Example: '{correct}'. "
                        f"Reason: {explanation}"
                    )
                    
                    all_contents.append(content)
                    # Add the source_id to the metadata for filtering
                    all_metadatas.append({"source": source_id})
                    all_ids.append(f"doc_{doc_id_counter}")
                    doc_id_counter += 1
                except (TypeError, AttributeError):
                    print(f"\n--- ERROR ---")
                    print(f"Invalid format in {filename}: Item '{item}' is not a valid dictionary or is missing keys.")
                    exit()

if not all_contents:
    print("No documents found to ingest.")
else:
    print(f"Found {len(all_contents)} total rules to process.")
    print("Generating embeddings...")
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=all_contents,
        task_type="RETRIEVAL_DOCUMENT"
    )
    embeddings = result['embedding']
    print("Embeddings generated.")

    collection.add(
        embeddings=embeddings,
        documents=all_contents,
        metadatas=all_metadatas,
        ids=all_ids
    )
    print(f"✅ Successfully ingested {collection.count()} rules into the database.")