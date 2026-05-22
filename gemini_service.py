import logging
from google import genai

log = logging.getLogger("notebook_lens")
client = genai.Client()

# =========================
# GEMINI API CALL
# =========================
def call_gemini(prompt):
    log.info(f"Gemini request sent | Prompt length: {len(prompt)} chars")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        log.info(f"Gemini response received | Response length: {len(response.text)} chars")
        return response.text
    except Exception as e:
        log.error(f"Gemini call failed | {type(e).__name__}: {e}")
        raise