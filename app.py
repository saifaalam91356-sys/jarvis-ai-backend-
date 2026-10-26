
import os
import logging
from flask import Flask, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

SYSTEM_INSTRUCTION = (
    "You are JARVIS, a friendly AI voice assistant. "
    "Reply naturally in Hindi or Hinglish. "
    "Be accurate, helpful and concise."
)


@app.get("/")
def home():
    return jsonify({
        "status": "ok",
        "service": "JARVIS AI backend"
    }), 200


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.post("/chat")
def chat():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return jsonify({
            "error": "GEMINI_API_KEY Render environment mein missing hai."
        }), 500

    data = request.get_json(silent=True) or {}
    message = data.get("message")

    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Message is required"}), 400

    try:
        client = genai.Client(api_key=api_key)

        app.logger.info("Gemini request started: model=%s", MODEL)

        response = client.models.generate_content(
            model=MODEL,
            contents=message.strip(),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                max_output_tokens=512
            )
        )

        reply = (response.text or "").strip()

        if not reply:
            app.logger.error("Gemini returned empty response")
            return jsonify({
                "error": "Gemini ne empty response diya."
            }), 502

        app.logger.info("Gemini reply received: %s characters", len(reply))

        return jsonify({"reply": reply}), 200

    except Exception:
        app.logger.exception("Gemini request failed")
        return jsonify({
            "error": "Gemini request failed. Render logs check karo."
        }), 502


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "10000"))
    )
