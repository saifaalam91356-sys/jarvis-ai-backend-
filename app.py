
import os
import logging
from flask import Flask, request, jsonify
from google import genai

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash"
]

SYSTEM_PROMPT = """
You are JARVIS AI, a friendly personal AI assistant.
Understand Hindi, Hinglish and English.
Reply naturally in the user's language.
Be helpful, accurate and concise.
You are running on the user's Android phone.
Do not claim to control the phone unless an action
has actually been performed by the Android application.
"""

@app.route("/", methods=["GET"])
def home():
    return "JARVIS AI backend is running!", 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "api_key_configured": bool(API_KEY)
    }), 200


@app.route("/chat", methods=["POST"])
def chat():
    if not API_KEY:
        return jsonify({
            "error": "GEMINI_API_KEY is not configured on Render."
        }), 500

    data = request.get_json(silent=True) or {}

    message = (
        data.get("message")
        or data.get("prompt")
        or data.get("text")
        or ""
    )

    if not isinstance(message, str) or not message.strip():
        return jsonify({
            "error": "Please provide a non-empty message."
        }), 400

    try:
        client = genai.Client(api_key=API_KEY)
    except Exception:
        logging.exception("Gemini client initialization failed")
        return jsonify({
            "error": "Could not initialize Gemini client."
        }), 500

    last_error = None

    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=(
                    SYSTEM_PROMPT
                    + "\n\nUser message:\n"
                    + message.strip()
                )
            )

            reply = response.text

            if not reply:
                raise ValueError("The model returned an empty response.")

            logging.info("Chat completed using %s", model)

            return jsonify({
                "reply": reply,
                "response": reply,
                "model": model
            }), 200

        except Exception as exc:
            last_error = exc
            logging.warning(
                "Model %s failed: %s", model, exc
            )

    logging.error("All Gemini models failed: %s", last_error)

    return jsonify({
        "error": "Gemini request failed. Check Render logs and API access."
    }), 502


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000"))
    )
