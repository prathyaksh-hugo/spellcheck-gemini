import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class GeminiClient:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def correct_text_with_brand_guide(self, text: str, language: str, context_rules: str) -> str:
        """
        Uses Gemini to correct text according to a detailed brand guide,
        and asks for a JSON object with the corrected text and a log of changes.
        """
        
        prompt = f"""
        You are an expert proofreader and Brand Voice Guardian for "Hugosave".
        Your task is to correct the "Original Text" by following a two-step process.

        **Step 1: General Proofreading**
        First, correct all general spelling and grammar errors using your expert knowledge of standard British English ({language}). This includes fixing typos in common words (e.g., 'appliction' -> 'application', 'mistike' -> 'mistake').

        **Step 2: Brand Guideline Alignment**
        After making general corrections, review the text and ensure it aligns perfectly with the specific "Brand Guidelines" provided below.

        **Brand Guidelines to Enforce:**
        {context_rules}

        **Additional Instructions:**
        - **Jargon Handling**: Recognize and preserve common technical or UX jargon (e.g., 'px' for pixels) even if it is not in the brand guide.
        - **List Handling**: If the 'Original Text' is a comma-separated list, correct each item individually and return the full, corrected, comma-separated list.
        - **Hashtag Rule**: Hashtags (words starting with #) must always be in lowercase.
        - **Output Format**: Your response MUST be a single, valid JSON object with two keys: `corrected_text` and `corrections_log`.
        - **Logging**: In the `corrections_log`, explain every change you made, whether it was a general correction or a brand-specific one.

        ---
        **Original Text:**
        "{text}"

        **Your JSON Response:**
        """
        try:
            generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
            response = self.model.generate_content(prompt, generation_config=generation_config)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return f'{{"corrected_text": "{text}", "corrections_log": ["Error processing request." ]}}'

    def correct_text(self, text: str, language: str = "en-GB") -> str:
        """(This is the older method, kept for reference but not used by the new SpellChecker)"""
        # ... (code for this method remains unchanged) ...
        return text