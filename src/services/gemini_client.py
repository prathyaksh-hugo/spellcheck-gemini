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

    def correct_batch_of_sentences(self, sentences: list[str], language: str, context_rules: str, check_type: str) -> str:
        """Processes a batch of sentences and categorizes the corrections based on the check type."""
        sentences_json_string = json.dumps(sentences, indent=2)
        
       
        analysis_instructions = ""
        correction_type = ""
        if check_type == "TYPO_BRAND":
            analysis_instructions = (
                "You are a Spelling and Brand Terminology Specialist. You will ONLY perform Typo & Brand Rule Analysis. Your primary directive is to find and correct spelling mistakes, punctuation errors, and brand rule violations based on the provided guidelines. These guidelines are the absolute source of truth and MUST override your general knowledge."
            )
            correction_type = "TYPO_BRAND"
        # This condition now exactly matches the 'check_type' from the /content-check endpoint
        elif check_type == "UX_WRITING": 
            analysis_instructions = (
                "You are a UX Writing and Style Editor. Your task is to rewrite sentences to improve clarity, tone, and grammatical structure, "
                "following the stylistic examples provided. Focus on punctuation (like the Oxford comma), tone of voice, and sentence flow. "
                "You should improve the text to make it more professional and user-friendly. You will ONLY perform Grammar & UX Writing Analysis. Your primary directive is to rewrite the sentence to improve its clarity, tone, and grammatical structure based on the provided guidelines."
            )
            correction_type = "UX_WRITING"

        prompt = f"""
        You are an expert proofreader with a specific role.
        
        **Your Role and Task:**
        {analysis_instructions}

        **Contextual Rules to Enforce (These are your source of truth):**
        {context_rules}

        After you have generated a `corrected_text`, you MUST compare it to the `original_text`.
        - If they are identical, you MUST set `is_correct` to `true` and the `corrections` array MUST be empty.
        - Only set `is_correct` to `false` if there is a genuine, visible change.

        **Additional Instructions:**
        - **Hashtag Rule**: Hashtags (words starting with #) must always be in lowercase, even if they contain a brand term. Example: '#hugoheroes' is correct, '#Hugoheroes' is incorrect.
        - **Do Not Flag Correct Terms**: If a brand term like 'Hugosave' or 'Hugosave Debit Card' is already spelled and capitalized correctly, do not flag it as an error.

        **Output Format:**
        Your response MUST be a single, valid JSON object with one key: "results".
        The value must be an array of objects, one for each input sentence. Each object must have:
        - "original_text": The original, unmodified sentence.
        - "is_correct": A boolean (true if you made NO changes, false otherwise).
        - "corrections": An array of objects detailing every single change. Each correction object must have:
            - "type": The category of the correction. This MUST be "{correction_type}".
            - "original": The specific word or phrase that was changed.
            - "suggestion": The corrected word or phrase.
        
        ---
        **Input Sentences (JSON Array):**
        {sentences_json_string}

        **Your Strict JSON Response:**
        """
        try:
            generation_config = genai.types.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.0 # Set to 0 for maximum consistency
            )
            response = self.model.generate_content(prompt, generation_config=generation_config)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return '{ "results": [] }'