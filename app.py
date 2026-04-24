"""
app.py — Flask web application for Image Captioning demo.
Run: python app.py
Then open: http://localhost:5000
"""

import os
import io
import base64
import torch
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
import config

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# ─── Globals loaded once ───────────────────────────────────────────────────────
_models = {}
_vocab  = None
_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _load_model(model_type: str):
    global _vocab
    from src.vocabulary import Vocabulary

    vocab_path = os.path.join(config.MODELS_DIR, "vocab.pkl")
    model_path = os.path.join(config.MODELS_DIR, f"{model_type}_best.pt")

    if _vocab is None:
        if not os.path.exists(vocab_path):
            return None, "Vocabulary not found. Please train the model first."
        _vocab = Vocabulary.load(vocab_path)

    if model_type not in _models:
        if not os.path.exists(model_path):
            return None, f"Model '{model_type}' not found. Train it with: python train_main.py --model {model_type}"
        if model_type == "lstm":
            from src.lstm_model import CaptioningLSTM
            m = CaptioningLSTM(vocab_size=len(_vocab))
        else:
            from src.transformer_model import CaptioningTransformer
            m = CaptioningTransformer(vocab_size=len(_vocab))
        m.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        m.to(config.DEVICE).eval()
        _models[model_type] = m

    return _models[model_type], None


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/caption", methods=["POST"])
def caption():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file       = request.files["image"]
    model_type = request.form.get("model", "lstm")
    method     = request.form.get("method", "greedy")

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    # Convert image to base64 for response
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    model, err = _load_model(model_type)
    if err:
        return jsonify({"error": err, "image": img_b64}), 200

    img_tensor = _transform(image).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        if method == "beam":
            from src.inference import beam_search
            cap = beam_search(model, img_tensor, _vocab)
        else:
            cap = model.caption(img_tensor, _vocab)[0]

    return jsonify({
        "caption": cap,
        "image"  : img_b64,
        "model"  : model_type,
        "method" : method,
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": config.DEVICE})


if __name__ == "__main__":
    print("\n🚀 Starting Image Captioning Web App")
    print(f"   Device : {config.DEVICE}")
    print(f"   URL    : http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
