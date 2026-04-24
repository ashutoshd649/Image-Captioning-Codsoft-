"""
src/train.py
Training loop with validation, learning-rate scheduling,
gradient clipping, and TensorBoard logging.
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import config


def train_one_epoch(model, loader, criterion, optimizer, device, clip=config.CLIP_GRAD):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for images, captions, lengths in tqdm(loader, desc="  Train", leave=False):
        images   = images.to(device)
        captions = captions.to(device)

        logits = model(images, captions)          # (B, T-1, V)
        targets = captions[:, 1:]                 # (B, T-1)

        # Flatten for CrossEntropyLoss
        V      = logits.size(-1)
        loss   = criterion(logits.reshape(-1, V), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        # Count non-pad tokens for accurate loss
        non_pad = (targets != 0).sum().item()
        total_loss   += loss.item() * non_pad
        total_tokens += non_pad

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for images, captions, lengths in tqdm(loader, desc="  Val  ", leave=False):
        images   = images.to(device)
        captions = captions.to(device)
        logits   = model(images, captions)
        targets  = captions[:, 1:]
        V        = logits.size(-1)
        loss     = criterion(logits.reshape(-1, V), targets.reshape(-1))
        non_pad  = (targets != 0).sum().item()
        total_loss   += loss.item() * non_pad
        total_tokens += non_pad

    return total_loss / max(total_tokens, 1)


def train(model, train_loader, val_loader, save_name="model",
          epochs=config.NUM_EPOCHS, lr=config.LEARNING_RATE, device=config.DEVICE):
    """
    Full training procedure.

    Returns:
        history : dict with keys 'train_loss', 'val_loss'
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)   # ignore <PAD>
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=3)

    writer    = SummaryWriter(log_dir=os.path.join(config.OUTPUTS_DIR, "runs", save_name))
    best_val  = float("inf")
    history   = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train={train_loss:.4f}  val={val_loss:.4f} | "
              f"{elapsed:.1f}s")

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(config.MODELS_DIR, f"{save_name}_best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved best model → {ckpt_path}")

    writer.close()
    # Also save final checkpoint
    torch.save(model.state_dict(),
               os.path.join(config.MODELS_DIR, f"{save_name}_final.pt"))
    print(f"\nTraining complete. Best val loss = {best_val:.4f}")
    return history
