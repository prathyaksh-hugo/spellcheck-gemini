# src/main.py
from importlib import metadata
import time
import json
import sqlite3
import re
import subprocess
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import all Pydantic models from their correct location
from .models.request_models import SpellCheckRequest, FeedbackRequest, IgnoreRequest
from .models.response_models import Metadata
from .services.gemini_client import GeminiClient
from .core.spell_checker import SpellChecker

# --- Constants and App Setup ---
KNOWLEDGE_BASE_PATH = "data/brand_guide_knowledge_base.json"
IGNORE_FILE_PATH = "data/ignore_list.txt"

app = FastAPI(
    title="Advanced Spell Checker AI",
    description="An intelligent spell and grammar checker using Gemini and RAG.",
    version="1.0.0"
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Service Initialization ---
gemini_client = GeminiClient()
spell_checker = SpellChecker(client=gemini_client)


# --- Helper Functions for Learning ---
def add_new_rule_to_knowledge_base(rule_content: str):
    """Adds a new learned rule to the JSON knowledge base and re-ingests."""
    try:
        with open(KNOWLEDGE_BASE_PATH, "r+") as f:
            knowledge_base = json.load(f)
            if any(rule["content"] == rule_content for rule in knowledge_base):
                print("Rule already exists in knowledge base.")
                return
            
            knowledge_base.append({
                "rule_type": "Formatting",
                "content": rule_content,
                "source": "Learned from user feedback"
            })
            f.seek(0)
            json.dump(knowledge_base, f, indent=2)
            f.truncate()
        
        print("Knowledge base updated. Re-running ingestion script...")
        subprocess.run(["python", "ingest.py"], check=True, capture_output=True, text=True)
        print("Ingestion complete. The AI has learned the new rule.")
    except Exception as e:
        print(f"Error updating knowledge base: {e}")

def add_to_simple_ignore_list(word: str):
    """Adds a word to the simple text-file ignore list."""
    try:
        with open(IGNORE_FILE_PATH, "r") as f:
            ignored_words = {line.strip().lower() for line in f}
    except FileNotFoundError:
        ignored_words = set()

    if word.lower() not in ignored_words:
        with open(IGNORE_FILE_PATH, "a") as f:
            f.write(f"{word}\n")
        return f"'{word}' added to simple ignore list."
    return f"'{word}' is already in the simple ignore list."


# --- API Endpoints ---
@app.post("/v1/spell-check")
async def spell_check(request: SpellCheckRequest):
    """Runs only the typo and brand rule check."""
    start_time = time.time()
    batch_results = spell_checker.batch_check_sentences(request.texts, "TYPO_BRAND")
    end_time = time.time()
    
  
    metadata = Metadata(
        total_processed=len(batch_results),
        processing_time_ms=int((end_time - start_time) * 1000),
        model_version="3.0.0-categorized"
    )
    return {"results": batch_results, "metadata": metadata}

@app.post("/v1/content-check")
async def content_check(request: SpellCheckRequest):
    """Runs only the grammar and UX writing check."""
    start_time = time.time()

    # --- THIS IS THE FIX ---
    # Filter the list to only include sentences with more than 3 words.
    texts_to_check = [
        sentence for sentence in request.texts if len(sentence.split()) > 3
    ]
    
    if not texts_to_check:
        # If no sentences meet the criteria, return an empty result immediately.
        return {"results": [], "metadata": {
            "total_processed": 0,
            "processing_time_ms": int((time.time() - start_time) * 1000),
            "model_version": "3.0.0-categorized"
        }}

    batch_results = spell_checker.batch_check_sentences(texts_to_check, "UX_WRITING")
    
    end_time = time.time()
    metadata = Metadata(
        total_processed=len(batch_results),
        processing_time_ms=int((end_time - start_time) * 1000),
        model_version="3.0.0-categorized"
    )
    return {"results": batch_results, "metadata": metadata}


@app.post("/v1/ignore")
async def handle_ignore_request(request: IgnoreRequest):
    """Saves ignored words and learns new patterns intelligently."""
    word_to_ignore = request.word.strip()
    if not word_to_ignore:
        return JSONResponse(status_code=400, content={"message": "Word cannot be empty."})

    # Use Advanced Learning ONLY for detectable patterns
    if re.match(r'^\S*\$\d+(\.\d+)?$', word_to_ignore):
        print(f"Detected currency pattern in '{word_to_ignore}'. Using advanced learning.")
        rule = f"Formatting Rule: Currency formats like '{word_to_ignore}' are valid."
        add_new_rule_to_knowledge_base(rule)
        status = f"Learned new currency pattern from '{word_to_ignore}'."
    else:
        # For EVERYTHING else, use the simple ignore list that the frontend reads
        print(f"'{word_to_ignore}' does not match a known pattern. Using simple ignore list.")
        status = add_to_simple_ignore_list(word_to_ignore)
    
    return {"status": status}

@app.get("/v1/ignore-list")
async def get_ignore_list():
    """Provides the simple ignore list to the frontend."""
    try:
        with open(IGNORE_FILE_PATH, "r") as f:
            ignored_words = [line.strip() for line in f]
        return {"ignore_list": ignored_words}
    except FileNotFoundError:
        return {"ignore_list": []}

@app.post("/v1/feedback", status_code=202)
async def submit_feedback(feedback: FeedbackRequest):
    """Receives and stores detailed user feedback on corrections."""
    conn = None
    try:
        conn = sqlite3.connect("feedback.db")
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO feedback (correction_id, action, original_text, corrected_text, suggested_text)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                feedback.correction_id,
                feedback.action,
                feedback.original_text,
                feedback.corrected_text,
                feedback.suggested_text,
            ),
        )
        conn.commit()
    finally:
        if conn:
            conn.close()
    
    return {"status": "Feedback received"}