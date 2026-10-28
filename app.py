from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

app = Flask(__name__)

# Retrieve model name from environment variables, fallback to Qwen2.5-0.5B
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B")

print(f"Loading model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
print("Model loaded successfully!")

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json or {}
    prompt = data.get("prompt", "")
    max_length = int(data.get("max_length", 512))
    
    if not prompt:
        return jsonify({"error": "Please provide a 'prompt' field in the JSON body."}), 400
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_length)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"response": generated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Render binds to the PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
