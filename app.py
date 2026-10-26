
import os
import logging
import time

from flask import Flask, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Render Environment variable se model names lo
MODELS = [
    m.strip()
    for m in os.environ.get(
        "GEMINI_MODELS",
        "gemini-3.8-flash"
    ).split(",")
    if m.strip()
]

SYSTEM_INSTRUCTION = (
    "You are JARVIS, a friendly AI voice assistant. "
    "Reply naturally in Hindi or Hinglish. "
    "Be helpful, accurate and concise."
)


@app.route("/", methods=["GET"])
def home():
    return "JARVIS AI backend is running!", 200


@app.route("/models", methods=["GET"])
def list_models():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY missing"}), 500

    try:
        client = genai.Client(api_key=api_key)
        available = []

        for model in client.models.list():
            actions = getattr(model, "supported_actions", []) or []
            if "generateContent" in actions:
                available.append(model.name.replace("models/", ""))

        return jsonify({"models": available}), 200

    except Exception:
        app.logger.exception("Could not list Gemini models")
        return jsonify({"error": "Could not list models"}), 500


@app.route("/chat", methods=["POST"])
def chat():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return jsonify({
            "error": "GEMINI_API_KEY Render par set nahi hai."
        }), 500

    data = request.get_json(silent=True) or {}
    message = data.get("message", "")

    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Message is required"}), 400

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
                            system_instruction=SYSTEM_INSTRUCTION,
                            max_output_tokens=1024
                        )
                    )

                    reply = response.text

                    if reply and reply.strip():
                        return jsonify({
                            "reply": reply.strip()
                        }), 200

                    last_error = "Empty response from Gemini"
                    break

                except Exception as e:
                    last_error = e
                    error_text = str(e).upper()

                    app.logger.exception(
                        "Model %s failed", model_name
                    )

                    temporary = any(code in error_text for code in (
                        "429", "503", "UNAVAILABLE",
                        "RESOURCE_EXHAUSTED", "500",
                        "INTERNAL SERVER ERROR"
                    ))

                    if temporary and attempt == 0:
                        time.sleep(3)
                        continue

                    # Invalid model/key errors par retry waste mat karo
                    break

        app.logger.error("All models failed: %s", str(last_error)[:300])

        return jsonify({
            "error": "Gemini AI abhi jawab nahi de pa raha.",
            "details": str(last_error)[:300]
        }), 503

    except Exception:
        app.logger.exception("Backend error")
        return jsonify({
            "error": "Backend request failed"
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
