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
        """Processes a whole batch of sentences in a single API call."""
        sentences_json_string = json.dumps(sentences)
        
        prompt = f"""
        You are an expert proofreader and Brand Voice Guardian. Your task is to process a JSON array of sentences.

        For each sentence, perform two steps:
        1. **General Proofreading**: Correct all spelling and grammar based on standard British English ({language}). Maintain correct sentence structure and capitalization (e.g., only capitalize the first word or proper nouns).
        2. **Brand Guideline Alignment**: Ensure the corrected sentence aligns with the provided "Brand Guidelines".

        **Brand Guidelines to Enforce:**
        {context_rules}

        **Additional Instructions:**
        - **Rule Precedence**: Brand Guidelines for proper nouns (e.g., 'Roundups') ALWAYS override standard English capitalization.
        - **Output Format**: Your response MUST be a single, valid JSON object with one key: "results". The value should be an array of objects, one for each input sentence. Each object must have:
          - "original_text": The original sentence.
          - "corrected_text": The corrected sentence.
          - "is_correct": A boolean (true if no changes were made).
          - "corrections_log": An array of objects detailing every change. Each log object must have "type", "original", and "corrected" keys.

        **Example Response Format:**
        {{
          "results": [
            {{
              "original_text": "please login to see your save account",
              "corrected_text": "Please Sign in to see your Save Account.",
              "is_correct": false,
              "corrections_log": [
                {{ "type": "Brand Rule", "original": "login", "corrected": "Sign in" }},
                {{ "type": "Capitalization", "original": "save account", "corrected": "Save Account" }}
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