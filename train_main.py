"""
train_main.py
CLI entry-point for training.

Usage:
    python train_main.py --model lstm        --epochs 20
    python train_main.py --model transformer --epochs 20 --lr 1e-4
"""

import argparse
import os
import torch
import config
from src.vocabulary import Vocabulary
from src.dataset import load_captions_df, build_dataloaders
from src.train import train


def parse_args():
    p = argparse.ArgumentParser(description="Train Image Captioning Model")
    p.add_argument("--model",   type=str,   default="lstm",
                   choices=["lstm", "transformer"], help="Model architecture")
    p.add_argument("--epochs",  type=int,   default=config.NUM_EPOCHS)
    p.add_argument("--lr",      type=float, default=config.LEARNING_RATE)
    p.add_argument("--batch",   type=int,   default=config.BATCH_SIZE)
    p.add_argument("--device",  type=str,   default=config.DEVICE)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\n{'='*55}")
    print(f"  Image Captioning Internship Project")
    print(f"  Model   : {args.model.upper()}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  LR      : {args.lr}")
    print(f"  Device  : {args.device}")
    print(f"{'='*55}\n")

    # ── 1. Load captions ──────────────────────────────────────
    if not os.path.exists(config.CAPTIONS_FILE):
        print(f"[ERROR] Captions file not found: {config.CAPTIONS_FILE}")
        print("  → Download Flickr8k from https://www.kaggle.com/datasets/adityajn105/flickr8k")
        print("  → Place captions.txt in the data/ folder")
        return

    print("Loading captions...")
    df = load_captions_df(config.CAPTIONS_FILE)
    print(f"  {len(df)} caption rows loaded")

    # ── 2. Build vocabulary ───────────────────────────────────
    vocab_path = os.path.join(config.MODELS_DIR, "vocab.pkl")
    if os.path.exists(vocab_path):
        vocab = Vocabulary.load(vocab_path)
    else:
        vocab = Vocabulary()
        vocab.build(df["caption"].tolist())
        vocab.save(vocab_path)

    # ── 3. Build data loaders ────────────────────────────────
    config.BATCH_SIZE = args.batch
    train_loader, val_loader, test_loader = build_dataloaders(
        df, vocab, config.IMAGES_DIR
    )
    print(f"  Train batches: {len(train_loader)} | "
          f"Val: {len(val_loader)} | Test: {len(test_loader)}\n")

    # ── 4. Build model ────────────────────────────────────────
    if args.model == "lstm":
        from src.lstm_model import CaptioningLSTM
        model = CaptioningLSTM(vocab_size=len(vocab))
    else:
        from src.transformer_model import CaptioningTransformer
        model = CaptioningTransformer(vocab_size=len(vocab))

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}\n")

    # ── 5. Train ──────────────────────────────────────────────
    history = train(
        model       = model,
        train_loader= train_loader,
        val_loader  = val_loader,
        save_name   = args.model,
        epochs      = args.epochs,
        lr          = args.lr,
        device      = args.device,
    )

    # ── 6. BLEU evaluation ────────────────────────────────────
    print("\nRunning BLEU evaluation on test set...")
    from src.evaluate import compute_bleu
    compute_bleu(model, test_loader, vocab, device=args.device)

    print("\n✓ Done! Check the models/ folder for saved checkpoints.")


if __name__ == "__main__":
    main()
