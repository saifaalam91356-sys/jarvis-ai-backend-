import os
import logging
import time

from flask import Flask, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Primary model aur fallback model
MODELS = [
    "gemini-3.8-flash",
    "gemini-2.5-flash"
]


@app.route("/", methods=["GET"])
def home():
    return "JARVIS AI backend is running!", 200


@app.route("/chat", methods=["POST"])
def chat():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        app.logger.error("GEMINI_API_KEY is missing")
        return jsonify({
            "error": "Gemini API key Render par set nahi hai."
        }), 500

    data = request.get_json(silent=True) or {}
    message = data.get("message", "")

    if not isinstance(message, str) or not message.strip():
        return jsonify({
            "error": "Message is required"
        }), 400

    try:
        client = genai.Client(api_key=api_key)
        last_error = None

        for model_name in MODELS:
            for attempt in range(2):
                try:
                    app.logger.info(
                        "Trying model %s, attempt %s",
                        model_name, attempt + 1
                    )

                    response = client.models.generate_content(
                        model=model_name,
                        contents=message.strip(),
                        config=types.GenerateContentConfig(
                            system_instruction=(
                                "You are JARVIS, a friendly AI voice "
                                "assistant. Reply naturally in Hindi "
                                "or Hinglish. Be helpful and concise."
                            )
                        )
                    )

                    reply = response.text

                    if reply and reply.strip():
                        return jsonify({
                            "reply": reply.strip()
                        }), 200

                    last_error = "Gemini returned an empty response"

                except Exception as e:
                    last_error = e

                    app.logger.exception(
                        "Model %s failed: %s",
                        model_name, type(e).__name__
                    )

                    error_text = str(e).upper()

                    if attempt == 0 and any(
                        code in error_text
                        for code in (
                            "503",
                            "UNAVAILABLE",
                            "429",
                            "RESOURCE_EXHAUSTED"
                        )
                    ):
                        time.sleep(2)
                        continue

                    break

        app.logger.error(
            "All Gemini models failed: %s",
            str(last_error)[:300]
        )

        return jsonify({
            "error": "Gemini AI abhi jawab nahi de pa raha.",
            "details": str(last_error)[:300]
        }), 503

    except Exception as e:
        app.logger.exception("Backend error")
        return jsonify({
            "error": "Backend request failed",
            "details": str(e)[:300]
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
