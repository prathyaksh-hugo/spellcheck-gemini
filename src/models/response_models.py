from pydantic import BaseModel, Field
from typing import List
import uuid 

class CorrectionLogEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str

class Suggestion(BaseModel):
    word: str
    confidence: float
    context_score: float

class CorrectionResult(BaseModel):
    original_text: str
    corrected_text: str
    is_correct: bool
    confidence: float
    error_type: str
    suggestions: List[Suggestion] = []
    # corrections_log: List[str] = []
    corrections_log: List[CorrectionLogEntry] = []

class Metadata(BaseModel):
    total_processed: int
    processing_time_ms: int
    model_version: str

class SpellCheckResponse(BaseModel):
    results: List[CorrectionResult]
    metadata: Metadata