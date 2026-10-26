
import os
import logging
from flask import Flask, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


@app.route("/", methods=["GET"])
def home():
    return "JARVIS AI backend is running!"


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
        return jsonify({"error": "Message is required"}), 400

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=message.strip(),
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are JARVIS, a friendly AI voice assistant. "
                    "Reply naturally in Hindi or Hinglish. "
                    "Be helpful and concise."
                )
            )
        )

        reply = response.text or "Maaf kijiye, abhi jawab nahi mila."
        return jsonify({"reply": reply})

    except Exception as e:
        app.logger.exception(
            "Gemini request failed: %s", type(e).__name__
        )
        return jsonify({
            "error": "Gemini AI request failed",
            "details": str(e)[:300]
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )

