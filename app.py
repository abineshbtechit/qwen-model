from flask import Flask, request, jsonify
import os
import time
import requests

app = Flask(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/free").strip()
OPENROUTER_FALLBACK_MODEL = os.environ.get("OPENROUTER_FALLBACK_MODEL", "").strip()
MAX_ATTEMPTS_PER_MODEL = 2


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
        return "".join(
            block["text"] for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ).strip()

    return ""


def provider_error_message(response):
    """Extract a useful provider error without exposing credentials."""
    try:
        body = response.json()
        error = body.get("error") or {}
        metadata = error.get("metadata") or {}
        message = metadata.get("raw") or error.get("message")
        return str(message)[:500] if message else "The inference provider returned an error."
    except (ValueError, AttributeError):
        return "The inference provider returned an unreadable error response."


def is_free_model(model):
    """Permit only the free router or model IDs explicitly marked free."""
    return model == "openrouter/free" or model.endswith(":free")


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

    if api_key.startswith("hf_"):
        return jsonify({
            "error": "The configured OPENROUTER_API_KEY looks like a Hugging Face token.",
            "hint": "Replace it in Render with a valid OpenRouter API key. Do not paste the key into chat."
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

    primary_model = OPENROUTER_MODEL or "openrouter/free"
    models_to_try = [primary_model]
    if OPENROUTER_FALLBACK_MODEL and OPENROUTER_FALLBACK_MODEL != primary_model:
        models_to_try.append(OPENROUTER_FALLBACK_MODEL)

    if any(not is_free_model(model) for model in models_to_try):
        return jsonify({
            "error": "Only explicitly free models are allowed by this API.",
            "hint": "Use openrouter/free or model IDs ending in :free for both model environment variables."
        }), 503

    attempts = []
    for model in models_to_try:
        for retry_number in range(1, MAX_ATTEMPTS_PER_MODEL + 1):
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt.strip()}],
                "max_tokens": max_tokens,
                "stream": False
            }

            try:
                response = requests.post(
                    OPENROUTER_URL, headers=headers, json=payload, timeout=90
                )

                if response.status_code >= 400:
                    attempts.append({
                        "model": model,
                        "try": retry_number,
                        "status": response.status_code,
                        "details": provider_error_message(response)
                    })
                    if response.status_code in (408, 429, 500, 502, 503, 504) and retry_number < MAX_ATTEMPTS_PER_MODEL:
                        time.sleep(1)
                        continue
                    break

                result = response.json()
                choices = result.get("choices") or []
                choice = choices[0] if choices else {}
                generated_text = extract_text(choice) if choice else ""
                finish_reason = choice.get("finish_reason") if choice else None

                if generated_text:
                    return jsonify({
                        "response": generated_text,
                        "model": result.get("model", model)
                    }), 200

                attempts.append({
                    "model": model,
                    "try": retry_number,
                    "status": response.status_code,
                    "details": "The provider returned no choices." if not choices else "The provider returned an empty response.",
                    "finish_reason": finish_reason,
                    "response_id": result.get("id")
                })
                if retry_number < MAX_ATTEMPTS_PER_MODEL:
                    time.sleep(1)

            except requests.Timeout:
                attempts.append({
                    "model": model, "try": retry_number, "status": None,
                    "details": "The inference provider timed out."
                })
                if retry_number < MAX_ATTEMPTS_PER_MODEL:
                    time.sleep(1)
            except requests.RequestException:
                attempts.append({
                    "model": model, "try": retry_number, "status": None,
                    "details": "Could not connect to the inference provider."
                })
                if retry_number < MAX_ATTEMPTS_PER_MODEL:
                    time.sleep(1)
            except (ValueError, TypeError, AttributeError):
                attempts.append({
                    "model": model, "try": retry_number, "status": None,
                    "details": "Received an invalid response from the inference provider."
                })
                break

    statuses = [attempt["status"] for attempt in attempts]
    if statuses and all(status == 429 for status in statuses):
        return jsonify({
            "error": "All configured models are currently rate-limited.",
            "hint": "Wait before retrying. Only free models are permitted by this API.",
            "attempts": attempts
        }), 429

    if 401 in statuses:
        return jsonify({
            "error": "OpenRouter or an upstream provider rejected authentication.",
            "hint": "Verify OPENROUTER_API_KEY in Render. Use an OpenRouter key, not a Hugging Face token.",
            "attempts": attempts
        }), 502

    return jsonify({
        "error": "No configured model returned usable text after retries.",
        "hint": "Inspect finish_reason and per-attempt details. Free model availability can fluctuate.",
        "attempts": attempts
    }), 502


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
