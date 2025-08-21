# src/core/spell_checker.py
import json
import chromadb
import google.generativeai as genai
from ..services.gemini_client import GeminiClient

# Define a character limit for each chunk to stay safely under the API's payload size limit.
CHARACTER_LIMIT_PER_CHUNK = 15000

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

    def _process_chunk(self, chunk: list[str]) -> list:
        """Helper function to process a single chunk and handle API calls."""
        if not chunk:
            return []
        
        relevant_rules = self._find_relevant_rules(chunk)
        response_str = self.client.correct_batch_of_sentences(
            chunk, "en-GB", relevant_rules
        )
        
        try:
            response_json = json.loads(response_str)
            return response_json.get("results", [])
        except json.JSONDecodeError:
            print(f"Failed to decode JSON from Gemini API for a chunk.")
            return []

    def batch_check_sentences(self, sentences: list[str]) -> list:
        """
        Checks a large list of sentences by dynamically creating chunks based on character count.
        """
        if not sentences:
            return []

        all_results = []
        current_chunk = []
        current_char_count = 0
        chunk_num = 1

        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding the next sentence would exceed the limit, process the current chunk.
            if current_chunk and (current_char_count + sentence_len > CHARACTER_LIMIT_PER_CHUNK):
                print(f"Processing chunk {chunk_num} with {len(current_chunk)} sentences ({current_char_count} chars)...")
                chunk_results = self._process_chunk(current_chunk)
                all_results.extend(chunk_results)
                
                # Reset for the next chunk
                current_chunk = []
                current_char_count = 0
                chunk_num += 1

            # Add the current sentence to the chunk
            current_chunk.append(sentence)
            current_char_count += sentence_len

        # Process the final remaining chunk after the loop finishes
        if current_chunk:
            print(f"Processing final chunk {chunk_num} with {len(current_chunk)} sentences ({current_char_count} chars)...")
            chunk_results = self._process_chunk(current_chunk)
            all_results.extend(chunk_results)

        return all_results