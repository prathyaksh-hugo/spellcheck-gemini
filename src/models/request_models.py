from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class SpellCheckRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="A list of texts to be checked.")
    language: str = Field("en-GB", description="Language standard to use.")
    # We will add other fields like confidence_threshold later

class FeedbackRequest(BaseModel):
    correction_id: str
    action: Literal["accept", "reject", "suggest"]
    original_text: str
    corrected_text: str
    suggested_text: str | None = None