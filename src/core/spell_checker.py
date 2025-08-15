# src/core/spell_checker.py
import json
import chromadb
import google.generativeai as genai
from ..services.gemini_client import GeminiClient

# Define a chunk size to keep requests under the API limit
CHUNK_SIZE = 25

class SpellChecker:
    def __init__(self, client: GeminiClient):
        self.client = client
        db_client = chromadb.PersistentClient(path="db")
        self.collection = db_client.get_collection(name="brand_voice_guide")

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

    def batch_check_sentences(self, sentences: list[str]) -> list:
        """
        Checks a large list of sentences by breaking it into smaller chunks
        to stay within the API's payload size limit.
        """
        if not sentences:
            return []

        all_results = []
        # Loop through the sentences in chunks of CHUNK_SIZE
        for i in range(0, len(sentences), CHUNK_SIZE):
            chunk = sentences[i:i + CHUNK_SIZE]
            print(f"Processing chunk {i//CHUNK_SIZE + 1} with {len(chunk)} sentences...")
            
            relevant_rules = self._find_relevant_rules(chunk)
            response_str = self.client.correct_batch_of_sentences(
                chunk, "en-GB", relevant_rules
            )
            
            try:
                response_json = json.loads(response_str)
                # Add the results from this chunk to our main list
                all_results.extend(response_json.get("results", []))
            except json.JSONDecodeError:
                print(f"Failed to decode JSON from Gemini API for chunk {i//CHUNK_SIZE + 1}")
                
        return all_results