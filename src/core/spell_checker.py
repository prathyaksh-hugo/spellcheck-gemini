# src/core/spell_checker.py
import json
import chromadb
import google.generativeai as genai
from ..services.gemini_client import GeminiClient

class SpellChecker:
    def __init__(self, client: GeminiClient):
        self.client = client
        db_client = chromadb.PersistentClient(path="db")
        self.collection = db_client.get_collection(name="unified_knowledge_base")
        print("SpellChecker initialized and connected to DB.")

    def _find_relevant_rules(self, text_list: list[str], source_id: str) -> str:
        if not text_list:
            return "No text provided for context search."
            
        combined_text = ", ".join(text_list)
        embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=combined_text,
            task_type="RETRIEVAL_QUERY"
        )['embedding']
        
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=15,
            where={"source": source_id} 
        )
        
        if not results or not results.get('documents') or not results['documents'][0]:
            print(f"Warning: No specific rules found in the database for source: '{source_id}'")
            return "No specific brand rules found."
            
        return "\n".join([f"- {doc}" for doc in results['documents'][0]])

    def _process_sentences(self, sentences: list[str], check_type: str) -> list:
        if not sentences:
            return []
        
        source_map = {
            "TYPO_BRAND": "spelling_and_terminology",
            "UX_WRITING": "grammar_and_style"
        }
        
        source_id = source_map.get(check_type)
        if not source_id:
            print(f"Error: Invalid check_type '{check_type}'. Cannot find relevant rules.")
            return []

        print(f"Finding relevant rules from source: '{source_id}'...")
        relevant_rules = self._find_relevant_rules(sentences, source_id)
        
        response_str = self.client.correct_batch_of_sentences(
            sentences, "en-GB", relevant_rules, check_type
        )
        
        try:
            response_json = json.loads(response_str)
            results = response_json.get("results", [])
            
            # ✅ Fix: Only mark is_correct = false if suggestion differs
            cleaned_results = []
            for r in results:
                original_text = r.get("original_text")
                corrections = r.get("corrections", [])

                valid_corrections = []
                for c in corrections:
                # 1. Reconstruct the full sentence for any incomplete UX_WRITING suggestions
                    if c.get("type") == "UX_WRITING" and c.get("original") != original_text:
                    # If the AI returned a word instead of the sentence, fix it.
                        c["suggestion"] = original_text.replace(c.get("original"), c.get("suggestion"))
                        c["original"] = original_text
                
                # 2. Ensure there's an actual, visible change before adding it
                    if c.get("suggestion") != c.get("original"):
                        valid_corrections.append(c)

            # 3. Build the final, clean result object
                    cleaned_results.append({
                    "original_text": original_text,
                    "is_correct": len(valid_corrections) == 0,
                    "corrections": valid_corrections
                })
        
            
            return cleaned_results

        except json.JSONDecodeError:
            print("Failed to decode JSON from Gemini API.")
            return []

    def batch_check_sentences(self, sentences: list[str], check_type: str) -> list:
        """Send all sentences at once (no chunking)."""
        if not sentences:
            return []
        
        print(f"Processing {len(sentences)} sentences at once...")
        return self._process_sentences(sentences, check_type)
