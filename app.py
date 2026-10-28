from flask import Flask, request, jsonify
import os
import requests

app = Flask(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/free")


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "inference_provider": "OpenRouter",
        "model": OPENROUTER_MODEL
    }), 200


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "")

    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify({
            "error": "Please provide a non-empty 'prompt' string in the JSON body."
        }), 400

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return jsonify({
            "error": "Server is missing the OPENROUTER_API_KEY environment variable."
        }), 503

    try:
        max_tokens = int(data.get("max_length", 512))
    except (TypeError, ValueError):
        return jsonify({"error": "'max_length' must be an integer."}), 400

    # Keep requests bounded and suitable for a small free-tier API.
    max_tokens = max(1, min(max_tokens, 2048))

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=90
        )

        if response.status_code >= 400:
            try:
                details = response.json().get("error", {}).get("message")
            except (ValueError, AttributeError):
                details = None

            if response.status_code == 429:
                return jsonify({
                    "error": "OpenRouter rate limit reached or the selected free model is busy. Please try again later.",
                    "provider_status": 429
                }), 429

            return jsonify({
                "error": details or "The inference provider returned an error.",
                "provider_status": response.status_code
            }), 502

        result = response.json()
        choices = result.get("choices", [])
        if not choices:
            return jsonify({"error": "The inference provider returned no choices."}), 502

        generated_text = choices[0].get("message", {}).get("content", "")
        return jsonify({
            "response": generated_text,
            "model": result.get("model", OPENROUTER_MODEL)
        }), 200

    except requests.Timeout:
        return jsonify({"error": "The inference provider took too long to respond. Please try again."}), 504
    except requests.RequestException:
        return jsonify({"error": "Could not connect to the inference provider."}), 502
    except (ValueError, KeyError, TypeError):
        return jsonify({"error": "Received an invalid response from the inference provider."}), 502


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
