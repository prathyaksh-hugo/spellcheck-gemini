# src/core/spell_checker.py
import json
import redis
import chromadb
import google.generativeai as genai
from ..services.gemini_client import GeminiClient
from ..models.response_models import CorrectionResult, CorrectionLogEntry

class SpellChecker:
    def __init__(self, client: GeminiClient):
        self.client = client
        db_client = chromadb.PersistentClient(path="db")
        self.collection = db_client.get_collection(name="brand_voice_guide")
        # The following line is commented out as per your request to disable caching
        # self.cache = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

    def _find_relevant_rules(self, text: str) -> str:
        """Queries ChromaDB to find relevant brand rules for the given text."""
        embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="RETRIEVAL_QUERY"
        )['embedding']
        
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=5
        )
        
        if not results['documents'] or not results['documents'][0]:
            return "No specific rules found."
            
        return "\n".join([f"- {doc}" for doc in results['documents'][0]])

    def check(self, original_text: str) -> CorrectionResult:
        """Checks text using the full brand guide RAG pipeline."""
        
        # Caching is disabled
        # cached_result = self.cache.get(original_text)
        # if cached_result: ...
        
        print(f"CACHE DISABLED for: '{original_text}'")
        relevant_rules = self._find_relevant_rules(original_text)
        response_str = self.client.correct_text_with_brand_guide(
            original_text, "en-GB", relevant_rules
        )
        
        log_from_llm = []
        try:
            response_json = json.loads(response_str)
            corrected_text = response_json.get("corrected_text", original_text)
            log_from_llm = response_json.get("corrections_log", [])
        except (json.JSONDecodeError, AttributeError):
            corrected_text = response_str 
            log_from_llm = ["Model did not return a valid JSON object."]

        is_correct = (original_text == corrected_text)
        
        # vvv THIS IS THE KEY CHANGE vvv
        # The AI may return strings or dicts. Convert everything to a string for safety.
        final_log = [CorrectionLogEntry(description=str(desc)) for desc in log_from_llm]

        final_result = CorrectionResult(
            original_text=original_text,
            corrected_text=corrected_text,
            is_correct=is_correct,
            confidence=0.95 if not is_correct else 1.0,
            error_type="style/grammar" if not is_correct else "none",
            corrections_log=final_log
        )
        
        # Caching is disabled
        # self.cache.set(original_text, final_result.model_dump_json(), ex=86400)
        
        return final_result