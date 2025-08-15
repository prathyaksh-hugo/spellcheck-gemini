import time
from fastapi import FastAPI
from .models.request_models import SpellCheckRequest, FeedbackRequest
from .models.response_models import SpellCheckResponse, Metadata
from .services.gemini_client import GeminiClient
from .core.spell_checker import SpellChecker

app = FastAPI(
    title="Advanced Spell Checker AI",
    description="An intelligent spell and grammar checker using Gemini and RAG.",
    version="1.0.0"
)

# Initialize services
gemini_client = GeminiClient()
spell_checker = SpellChecker(client=gemini_client)

@app.get("/health", status_code=200)
def health_check():
    """Health check endpoint to ensure the service is running."""
    return {"status": "ok"}

@app.post("/v1/check", response_model=SpellCheckResponse)
async def check_spelling(request: SpellCheckRequest):
    """
    Processes a batch of texts for spelling and grammar correction.
    """
    start_time = time.time()

    results = [spell_checker.check(text) for text in request.texts]

    end_time = time.time()
    processing_time_ms = int((end_time - start_time) * 1000)

    metadata = Metadata(
        total_processed=len(results),
        processing_time_ms=processing_time_ms,
        model_version="1.0.0-mvp"
    )

    return SpellCheckResponse(results=results, metadata=metadata)


@app.post("/v1/feedback", status_code=202)
async def submit_feedback(feedback: FeedbackRequest):
    """Receives and stores user feedback on corrections."""
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