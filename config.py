"""
config.py — Central configuration for the Image Captioning project.
Edit this file to change paths, hyperparameters, and model settings.
"""

import os

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data")
IMAGES_DIR     = os.path.join(DATA_DIR, "Images")
CAPTIONS_FILE  = os.path.join(DATA_DIR, "captions.txt")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR    = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ─── Vocabulary ────────────────────────────────────────────────────────────────
VOCAB_SIZE       = 5000    # Keep only top-N most frequent words
MIN_WORD_FREQ    = 5       # Ignore words appearing fewer than this many times
MAX_CAPTION_LEN  = 35      # Maximum caption token length

# Special tokens
PAD_TOKEN   = "<PAD>"
START_TOKEN = "<START>"
END_TOKEN   = "<END>"
UNK_TOKEN   = "<UNK>"

# ─── Feature Extraction ────────────────────────────────────────────────────────
ENCODER_MODEL    = "resnet50"   # "resnet50" | "vgg16"
IMAGE_SIZE       = 224          # Input image size for CNN
FEATURE_DIM      = 2048         # ResNet50 output dim (512 for VGG16)
FINE_TUNE_ENCODER = False       # Fine-tune CNN weights

# ─── LSTM Model ────────────────────────────────────────────────────────────────
EMBED_DIM    = 256
HIDDEN_DIM   = 512
NUM_LAYERS   = 2
DROPOUT      = 0.5

# ─── Transformer Model ─────────────────────────────────────────────────────────
D_MODEL      = 512
NHEAD        = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD    = 2048
TRANSFORMER_DROPOUT = 0.1

# ─── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS    = 20
CLIP_GRAD     = 5.0
WEIGHT_DECAY  = 1e-4

# Dataset splits
TRAIN_SPLIT = 0.80
VAL_SPLIT   = 0.10
TEST_SPLIT  = 0.10

# ─── Inference ─────────────────────────────────────────────────────────────────
BEAM_SIZE         = 5
MAX_DECODE_LEN    = 40

# ─── Device ────────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
