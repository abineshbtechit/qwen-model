from flask import Flask, request, jsonify
import os
import requests

app = Flask(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/free").strip()
# Set this to a DIFFERENT available OpenRouter model ID in Render.
# Leave empty to disable fallback rather than silently selecting an unknown model.
OPENROUTER_FALLBACK_MODEL = os.environ.get("OPENROUTER_FALLBACK_MODEL", "").strip()


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "inference_provider": "OpenRouter",
        "model": OPENROUTER_MODEL,
        "fallback_configured": bool(
            OPENROUTER_FALLBACK_MODEL
            and OPENROUTER_FALLBACK_MODEL != OPENROUTER_MODEL
        )
    }), 200


def extract_text(choice):
    """Extract generated text from an OpenAI-compatible chat completion."""
    message = choice.get("message") or {}
    content = message.get("content")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts).strip()

    return ""


def provider_error_message(response):
    """Extract a useful provider error without exposing credentials."""
    try:
        body = response.json()
        error = body.get("error") or {}
        metadata = error.get("metadata") or {}
        return (
            metadata.get("raw")
            or error.get("message")
            or "The inference provider returned an error."
        )
    except (ValueError, AttributeError):
        return "The inference provider returned an unreadable error response."


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "")

    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify({
            "error": "Please provide a non-empty 'prompt' string in the JSON body."
        }), 400

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return jsonify({
            "error": "Server is missing the OPENROUTER_API_KEY environment variable.",
            "hint": "Add a valid OpenRouter API key in Render Environment settings."
        }), 503

    # Hugging Face tokens (hf_...) cannot authenticate to OpenRouter.
    if api_key.startswith("hf_"):
        return jsonify({
            "error": "The configured OPENROUTER_API_KEY looks like a Hugging Face token.",
            "hint": "Replace it in Render with a valid OpenRouter API key from https://openrouter.ai/settings/keys. Do not paste the key into chat."
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

    # Only try distinct, explicitly configured model IDs.
    models_to_try = [OPENROUTER_MODEL or "openrouter/free"]
    if (
        OPENROUTER_FALLBACK_MODEL
        and OPENROUTER_FALLBACK_MODEL not in models_to_try
    ):
        models_to_try.append(OPENROUTER_FALLBACK_MODEL)

    attempts = []

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
                attempts.append({
                    "model": model,
                    "status": response.status_code,
                    "details": provider_error_message(response)
                })
                continue

            result = response.json()
            choices = result.get("choices") or []
            if not choices:
                attempts.append({
                    "model": model,
                    "status": response.status_code,
                    "details": "The provider returned no choices."
                })
                continue

            generated_text = extract_text(choices[0])
            if not generated_text:
                attempts.append({
                    "model": model,
                    "status": response.status_code,
                    "details": "The provider returned an empty response."
                })
                continue

            return jsonify({
                "response": generated_text,
                "model": result.get("model", model)
            }), 200

        except requests.Timeout:
            attempts.append({
                "model": model,
                "status": None,
                "details": "The inference provider timed out."
            })
        except requests.RequestException:
            attempts.append({
                "model": model,
                "status": None,
                "details": "Could not connect to the inference provider."
            })
        except (ValueError, TypeError, AttributeError):
            attempts.append({
                "model": model,
                "status": None,
                "details": "Received an invalid response from the inference provider."
            })

    statuses = [attempt["status"] for attempt in attempts]
    # Report 429 only when every attempted model was rate-limited.
    if statuses and all(status == 429 for status in statuses):
        return jsonify({
            "error": "All configured models are currently rate-limited.",
            "hint": "Wait and retry, or configure a different available model in OPENROUTER_FALLBACK_MODEL.",
            "attempts": attempts
        }), 429

    # Keep upstream auth failures distinct from rate limits and other errors.
    if 401 in statuses:
        return jsonify({
            "error": "OpenRouter or an upstream provider rejected authentication.",
            "hint": "Verify OPENROUTER_API_KEY in Render and inspect the per-model details below. Use an OpenRouter key, not a Hugging Face token.",
            "attempts": attempts
        }), 502

    return jsonify({
        "error": "No configured model returned usable text.",
        "hint": "Check the per-model details. Configure a different available model as OPENROUTER_FALLBACK_MODEL if needed.",
        "attempts": attempts
    }), 502


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
