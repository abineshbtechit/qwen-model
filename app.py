from flask import Flask, request, jsonify
import os
import requests

app = Flask(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/free")
OPENROUTER_FALLBACK_MODEL = os.environ.get(
    "OPENROUTER_FALLBACK_MODEL",
    "google/gemma-3-4b-it:free"
)


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "inference_provider": "OpenRouter",
        "model": OPENROUTER_MODEL
    }), 200


def extract_text(choice):
    """Extract generated text from an OpenAI-compatible chat completion."""
    message = choice.get("message") or {}
    content = message.get("content")

    if isinstance(content, str):
        return content.strip()

    # Some providers return content as a list of text blocks.
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts).strip()

    return ""


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

    max_tokens = max(1, min(max_tokens, 2048))
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # If the router returns an empty completion, try a second free model.
    models_to_try = [OPENROUTER_MODEL]
    if OPENROUTER_FALLBACK_MODEL and OPENROUTER_FALLBACK_MODEL not in models_to_try:
        models_to_try.append(OPENROUTER_FALLBACK_MODEL)

    last_provider_status = None
    last_error = None

    for model in models_to_try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt.strip()}],
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=90
            )

            if response.status_code >= 400:
                last_provider_status = response.status_code
                try:
                    provider_error = response.json().get("error", {}).get("message")
                except (ValueError, AttributeError):
                    provider_error = None

                last_error = provider_error or "The inference provider returned an error."

                if response.status_code == 429:
                    continue
                continue

            result = response.json()
            choices = result.get("choices") or []
            if not choices:
                last_error = "The provider returned no choices."
                continue

            generated_text = extract_text(choices[0])
            if not generated_text:
                last_error = "The provider returned an empty response."
                continue

            return jsonify({
                "response": generated_text,
                "model": result.get("model", model)
            }), 200

        except requests.Timeout:
            last_error = "The inference provider took too long to respond."
            continue
        except requests.RequestException:
            last_error = "Could not connect to the inference provider."
            continue
        except (ValueError, TypeError, AttributeError):
            last_error = "Received an invalid response from the inference provider."
            continue

    if last_provider_status == 429:
        return jsonify({
            "error": "OpenRouter rate limit reached or the available free models are busy. Please try again later.",
            "provider_status": 429
        }), 429

    return jsonify({
        "error": "No free model returned usable text. Please try again later.",
        "details": last_error or "The inference provider returned no usable text.",
        "provider_status": last_provider_status
    }), 502


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
