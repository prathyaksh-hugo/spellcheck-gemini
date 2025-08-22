# src/core/spell_checker.py
import json
import chromadb
import google.generativeai as genai
from ..services.gemini_client import GeminiClient

class SpellChecker:
    def __init__(self, client: GeminiClient):
        self.client = client
        db_client = chromadb.PersistentClient(path="db")
        self.collection = db_client.get_collection(name="unified_knowledge_base") # Corrected collection name

    def _find_relevant_rules(self, text_list: list[str]) -> str:
        """Queries ChromaDB to find relevant rules for a list of texts."""
        combined_text = ", ".join(text_list)
        embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=combined_text,
            task_type="RETRIEVAL_QUERY"
        )['embedding']
        
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=15
        )
        
        if not results['documents'] or not results['documents'][0]:
            return "No specific rules found."
            
        return "\n".join([f"- {doc}" for doc in results['documents'][0]])

    def _process_chunk(self, chunk: list[str]) -> list:
        """Helper function to process a single chunk and handle API calls."""
        if not chunk:
            return []
        
        relevant_rules = self._find_relevant_rules(chunk)
        response_str = self.client.correct_batch_of_sentences(
            chunk, "en-GB", relevant_rules
        )
        print(f"--- RAW RESPONSE FROM GEMINI ---\n{response_str}\n---------------------------------")
        try:
            response_json = json.loads(response_str)
            return response_json.get("results", [])
        except json.JSONDecodeError:
            print(f"Failed to decode JSON from Gemini API for a chunk.")
            return []

    def batch_check_sentences(self, sentences: list[str]) -> list:
        """
        Checks a list of sentences in a single batch without chunking.
        WARNING: This may fail if the total character count is too high.
        """
        if not sentences:
            return []

        print(f"Processing all {len(sentences)} sentences in a single batch...")
        
        # Directly process the entire list as one chunk
        all_results = self._process_chunk(sentences)

        return all_results