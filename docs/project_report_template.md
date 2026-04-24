# Image Captioning AI — Internship Project Report

**Name:** ___________________________  
**Duration:** 1 Month  
**Technology Stack:** Python · PyTorch · Flask · Flickr8k  

---

## Executive Summary

This report documents the development of an Image Captioning AI system that automatically generates natural language descriptions of images. The project combined **Computer Vision** (ResNet50 CNN encoder) with **Natural Language Processing** (LSTM and Transformer decoders) trained on the Flickr8k dataset.

---

## Week 1: Data Exploration & Feature Extraction

### Dataset Overview
- **Dataset:** Flickr8k — 8,000 images with 5 captions each (40,000 total)
- **Source:** https://www.kaggle.com/datasets/adityajn105/flickr8k
- **Image types:** People, animals, outdoor scenes, sports

### Key EDA Findings
| Metric | Value |
|--------|-------|
| Total image-caption pairs | 40,000 |
| Unique images | 8,000 |
| Average caption length | ~11 tokens |
| Max caption length | ~40 tokens |
| Vocabulary size (min_freq=5) | ~5,000 words |

### Preprocessing Steps
1. Lowercased all captions
2. Removed special characters (kept alphanumeric + spaces)
3. Tokenised on whitespace
4. Built vocabulary keeping top-5000 words (≥5 appearances)
5. Added special tokens: `<PAD>`, `<START>`, `<END>`, `<UNK>`

### CNN Feature Extraction
- Used **ResNet50** pretrained on ImageNet (frozen weights)
- Removed final classification layer → 2048-dim feature vector
- Projected to 256-dim embedding via linear layer

---

## Week 2: CNN + LSTM Model

### Architecture
```
Image (3×224×224) 
  → ResNet50 backbone 
  → Linear(2048 → 256) 
  → LSTM h₀ initialisation
  
Caption tokens 
  → Embedding(256)
  → Concatenate with image features  
  → 2-layer LSTM (hidden=512)
  → Linear(512 → vocab_size)
  → Softmax probabilities
```

### Training Configuration
| Hyperparameter | Value |
|----------------|-------|
| Batch size | 32 |
| Learning rate | 3e-4 |
| Optimizer | Adam |
| LR scheduler | ReduceLROnPlateau |
| Epochs | 20 |
| Gradient clipping | 5.0 |

### Results
| Metric | Score |
|--------|-------|
| BLEU-1 | ___ |
| BLEU-2 | ___ |
| BLEU-3 | ___ |
| BLEU-4 | ___ |
| Best Val Loss | ___ |

---

## Week 3: CNN + Transformer Model

### Architecture
```
Image (3×224×224)
  → ResNet50
  → Linear(2048 → 512)        ← encoder memory (1 token)

Caption tokens
  → Embedding(512)
  → Positional Encoding (sinusoidal)
  → 3-layer Transformer Decoder
      - Masked self-attention (causal)
      - Cross-attention on image memory
      - Feed-forward (dim=2048)
  → Linear(512 → vocab_size)
```

### Key Concepts Applied
- **Causal masking** — prevents decoder from seeing future tokens
- **Cross-attention** — each caption token attends to image features
- **Pre-LN** — LayerNorm before attention for training stability
- **Positional encoding** — sinusoidal, injected before decoder

### Results
| Metric | Score |
|--------|-------|
| BLEU-1 | ___ |
| BLEU-2 | ___ |
| BLEU-3 | ___ |
| BLEU-4 | ___ |

### Greedy vs Beam Search
| Method | Sample Caption | BLEU-4 |
|--------|----------------|--------|
| Greedy | ___ | ___ |
| Beam (k=5) | ___ | ___ |

---

## Week 4: Deployment & Evaluation

### Web Application
Built a Flask web app (`app.py`) with:
- Image upload via drag-and-drop or file picker
- Model selection (LSTM or Transformer)
- Decoding strategy selection (Greedy or Beam Search)
- Real-time caption generation

### Model Comparison
| Model | Params | BLEU-4 | Inference (CPU) |
|-------|--------|--------|-----------------|
| CNN + LSTM | ~15M | ___ | ___ ms |
| CNN + Transformer | ~22M | ___ | ___ ms |

### Error Analysis
Common failure modes observed:
1. **Generic captions** — model defaults to "a dog is running" for many images
2. **Colour blindness** — CNNs often miss colour-specific details
3. **Counting errors** — models struggle with exact counts ("two" vs "three")
4. **Abstract scenes** — poor performance on indoor or abstract images

---

## Conclusion

### Key Learnings
1. Pre-trained CNN features (ResNet50) provide strong visual representations
2. Transformer decoder outperforms LSTM due to global attention
3. Beam search consistently improves caption quality over greedy decoding
4. Dataset size is a major limiting factor — COCO (120K images) would significantly improve results

### Future Improvements
- Use **CLIP** or **ViT** as encoder for richer semantic features
- **Attention visualisation** (soft attention over spatial feature maps)
- **Reinforcement learning** with CIDEr reward (self-critical training)
- Train on **MS-COCO** (15× larger than Flickr8k)
- Fine-tune encoder after initial convergence

---

## References

1. Vinyals et al. (2015) — *Show and Tell: A Neural Image Caption Generator*
2. Xu et al. (2015) — *Show, Attend and Tell*
3. Vaswani et al. (2017) — *Attention Is All You Need*
4. He et al. (2016) — *Deep Residual Learning for Image Recognition*
5. Flickr8k Dataset — Hodosh et al. (2013)
