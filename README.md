# 🖼️ Image Captioning AI — 1 Month Internship Project

> Combines **Computer Vision** (ResNet/VGG) + **NLP** (LSTM/Transformer) to auto-generate image captions.

---

## 📅 Month-wise Weekly Plan

| Week | Topic | Files |
|------|-------|-------|
| Week 1 | EDA, data preprocessing, feature extraction | `notebooks/Week1_EDA_and_Features.ipynb` |
| Week 2 | LSTM-based caption model (train + eval) | `notebooks/Week2_LSTM_Model.ipynb` |
| Week 3 | Transformer-based caption model | `notebooks/Week3_Transformer_Model.ipynb` |
| Week 4 | Flask web app + final demo | `notebooks/Week4_WebApp_and_Demo.ipynb` |

---

## 🏗️ Project Structure

```
image_captioning_internship/
├── README.md
├── requirements.txt
├── setup.py
├── config.py                   # All hyperparameters & paths
│
├── src/
│   ├── __init__.py
│   ├── dataset.py              # Flickr8k / COCO dataset loader
│   ├── feature_extractor.py    # ResNet50 / VGG16 feature extraction
│   ├── vocabulary.py           # Build & manage vocabulary
│   ├── lstm_model.py           # CNN+LSTM model
│   ├── transformer_model.py    # CNN+Transformer model
│   ├── train.py                # Training loop
│   ├── evaluate.py             # BLEU score evaluation
│   └── inference.py            # Caption generation (greedy + beam search)
│
├── notebooks/
│   ├── Week1_EDA_and_Features.ipynb
│   ├── Week2_LSTM_Model.ipynb
│   ├── Week3_Transformer_Model.ipynb
│   └── Week4_WebApp_and_Demo.ipynb
│
├── templates/
│   └── index.html              # Flask web UI
├── static/
│   ├── css/style.css
│   └── js/app.js
│
├── app.py                      # Flask web application
├── train_main.py               # CLI training script
├── tests/
│   └── test_all.py
└── docs/
    └── project_report_template.docx
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Flickr8k Dataset
- Visit: https://www.kaggle.com/datasets/adityajn105/flickr8k
- Extract into `data/` folder so you have:
  - `data/Images/` (8000 images)
  - `data/captions.txt`

### 3. Train the Model
```bash
# Train LSTM model
python train_main.py --model lstm --epochs 20

# Train Transformer model
python train_main.py --model transformer --epochs 20
```

### 4. Run Web App
```bash
python app.py
# Open http://localhost:5000
```

### 5. Generate Caption for Single Image
```python
from src.inference import generate_caption
caption = generate_caption("path/to/image.jpg", model_type="lstm")
print(caption)
```

---

## 📊 Expected Results

| Model | BLEU-1 | BLEU-4 | Training Time |
|-------|--------|--------|---------------|
| CNN + LSTM | ~0.60 | ~0.25 | ~2 hrs (GPU) |
| CNN + Transformer | ~0.65 | ~0.30 | ~3 hrs (GPU) |

---

## 🔧 Key Technologies
- **PyTorch** — deep learning framework
- **torchvision** — pre-trained ResNet50/VGG16
- **NLTK** — BLEU score computation
- **Flask** — web application
- **Pillow** — image processing
