# qwen-model

A small Flask API that generates text through OpenRouter's OpenAI-compatible chat completions API.

## Endpoints

### Health check

`GET /health`

Returns service status, the configured primary model, and whether a distinct fallback is configured. This endpoint does not test whether the provider key or model is currently usable.

### Generate text

`POST /generate`

JSON body:

```json
{
  "prompt": "Explain cloud computing in simple words.",
  "max_length": 150
}
```

A successful response contains `response` and `model`.

## Render environment variables

Configure these in your Render service's **Environment** settings:

- `OPENROUTER_API_KEY` — a valid key created at https://openrouter.ai/settings/keys. Do not use a Hugging Face token (`hf_...`) here.
- `OPENROUTER_MODEL` — primary OpenRouter model ID. Example: `google/gemma-4-26b-a4b-it:free`.
- `OPENROUTER_FALLBACK_MODEL` — optional, different model ID to try if the primary model fails. Choose a currently available model from https://openrouter.ai/collections/free-models. If empty, fallback is disabled.

Do not commit API keys to this repository or share them in screenshots.

## Rate limits and errors

- `400`: invalid or missing prompt / max_length.
- `503`: missing key or a Hugging Face token was mistakenly configured as the OpenRouter key.
- `429`: every configured model was rate-limited.
- `502`: authentication/provider errors or no configured model returned usable text. The response includes per-model status and details to help debugging.

A fallback can help when one model is unavailable, but it cannot guarantee service if all configured providers are rate-limited. A free model's shared provider capacity is controlled by the provider, not by this application.

## Local development

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python app.py
```

The app listens on the `PORT` environment variable or port `10000` by default.
