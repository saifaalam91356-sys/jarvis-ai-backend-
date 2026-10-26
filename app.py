
import os
import logging
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route("/", methods=["GET"])
def home():
    return "JARVIS AI backend is running!"

@app.route("/chat", methods=["POST"])
def chat():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        app.logger.error("OPENAI_API_KEY is missing")
        return jsonify({"error": "Server API key configured nahi hai."}), 500

    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Message is required"}), 400

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are JARVIS, a friendly voice assistant. "
                        "Reply naturally in Hindi or Hinglish."
                    )
                },
                {"role": "user", "content": message}
            ]
        )
        return jsonify({"reply": response.output_text})

    except Exception as e:
        app.logger.exception(
            "OpenAI request failed: %s", type(e).__name__
        )
        return jsonify({
            "error": "AI request failed",
            "details": str(e)[:500]
        }), 500

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )

