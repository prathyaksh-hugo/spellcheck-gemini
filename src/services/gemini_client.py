# src/services/gemini_client.py
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class GeminiClient:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')

    def correct_batch_of_sentences(self, sentences: list[str], language: str, context_rules: str) -> str:
        """Processes a batch of sentences and categorizes the corrections."""
        sentences_json_string = json.dumps(sentences)
    
        prompt = f"""
        You are an expert proofreader and Brand Voice Guardian. Your task is to process a JSON array of sentences.
        For each sentence, you will perform two types of analysis:

        1.  **Typo & Brand Rule Analysis**: Check for simple errors like spelling mistakes, punctuation, and capitalization based on the provided "Brand Guidelines".
        2.  **Grammar & UX Writing Analysis**: Rewrite the sentence to improve its clarity, tone, and grammatical structure, making it more professional and user-friendly.

        **Brand Guidelines to Enforce:**
        {context_rules}

        **Output Format:**
        Your response MUST be a single, valid JSON object with one key: "results".
        The value should be an array of objects, one for each input sentence. Each object must have:
        - "original_text": The original sentence.
        - "is_correct": A boolean (true if NO changes of any kind were made).
        - "corrections": An array of objects detailing every change. Each correction object must have:
            - "type": The category of the correction. Must be either "TYPO_BRAND" or "UX_WRITING".
            - "original": The specific word or phrase that was changed.
            - "suggestion": The corrected text.

        **Example Response:**
        {{
          "results": [
            {{
              "original_text": "manige your cash accountt",
              "is_correct": false,
              "corrections": [
                {{ "type": "TYPO_BRAND", "original": "manige", "suggestion": "manage" }},
                {{ "type": "TYPO_BRAND", "original": "accountt", "suggestion": "Account" }},
                {{ "type": "UX_WRITING", "original": "manige your cash accountt", "suggestion": "Manage your Cash Account." }}
              ]
            }}
          ]
        }}
        
        ---
        **Original Sentences (JSON Array):**
        {sentences_json_string}

        **Your JSON Response:**
        """
        try:
            generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
            response = self.model.generate_content(prompt, generation_config=generation_config)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return '{ "results": [] }'