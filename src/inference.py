"""
src/inference.py
Caption generation helpers:
  • greedy decoding  (fast)
  • beam search      (better quality)
  • generate_caption (convenience wrapper — loads model from disk)
"""

import os
import torch
from PIL import Image
from torchvision import transforms
import config
from src.vocabulary import Vocabulary


# ─── Image pre-processing ──────────────────────────────────────────────────────
_TRANSFORM = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225]),
])


def load_image(image_path: str, device: str = config.DEVICE) -> torch.Tensor:
    """Load an image from disk and return a (1, 3, H, W) tensor."""
    img = Image.open(image_path).convert("RGB")
    return _TRANSFORM(img).unsqueeze(0).to(device)


# ─── Beam Search ───────────────────────────────────────────────────────────────
def beam_search(model, image_tensor: torch.Tensor, vocab,
                beam_size: int = config.BEAM_SIZE,
                max_len: int   = config.MAX_DECODE_LEN) -> str:
    """
    Beam search decoding for a single image.

    Args:
        model        : trained CaptioningLSTM or CaptioningTransformer
        image_tensor : (1, 3, H, W)
        vocab        : Vocabulary instance
        beam_size    : number of beams
        max_len      : maximum generation length

    Returns:
        best_caption : str
    """
    model.eval()
    device = image_tensor.device

    with torch.no_grad():
        # Get image features
        img_features = model.encoder(image_tensor)   # (1, embed/d_model)

        start_tok = vocab.stoi[config.START_TOKEN]
        end_tok   = vocab.stoi[config.END_TOKEN]

        # Each beam: (log_prob, token_list)
        beams = [(0.0, [start_tok])]

        for _ in range(max_len):
            candidates = []
            for log_prob, tokens in beams:
                if tokens[-1] == end_tok:
                    candidates.append((log_prob, tokens))
                    continue

                tok_tensor = torch.tensor([tokens], dtype=torch.long, device=device)

                # Forward pass (handles both LSTM and Transformer)
                if hasattr(model, "decoder") and hasattr(model.decoder, "lstm"):
                    # LSTM model: step-by-step
                    h, c = model.decoder.init_hidden(img_features)
                    for t_idx in range(len(tokens)):
                        w_emb   = model.decoder.embed(tok_tensor[:, t_idx].unsqueeze(1))
                        img_exp = img_features.unsqueeze(1)
                        inp     = torch.cat([w_emb, img_exp], dim=-1)
                        _, (h, c) = model.decoder.lstm(inp, (h, c))
                    logits = model.decoder.fc(h[-1])        # (1, V)
                else:
                    # Transformer model
                    memory = img_features.unsqueeze(1)
                    logits = model.decoder(memory, tok_tensor)  # (1, T, V)
                    logits = logits[:, -1, :]                   # (1, V)

                log_probs = torch.log_softmax(logits.squeeze(0), dim=-1)
                top_probs, top_ids = log_probs.topk(beam_size)

                for p, idx in zip(top_probs.tolist(), top_ids.tolist()):
                    candidates.append((log_prob + p, tokens + [idx]))

            # Keep top-k beams
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]

            # Early stop if all beams ended
            if all(b[1][-1] == end_tok for b in beams):
                break

        # Best beam (highest log-prob, completed or longest)
        best_tokens = max(beams, key=lambda x: x[0])[1]
        return vocab.decode(best_tokens[1:])   # skip <START>


# ─── Convenience function ──────────────────────────────────────────────────────
def generate_caption(image_path: str,
                     model_type : str = "lstm",
                     method     : str = "beam",
                     device     : str = config.DEVICE) -> str:
    """
    High-level API: load saved model + vocab and caption an image.

    Args:
        image_path : path to image file
        model_type : "lstm" | "transformer"
        method     : "greedy" | "beam"
        device     : "cpu" | "cuda"

    Returns:
        caption string
    """
    from src.vocabulary import Vocabulary

    vocab_path = os.path.join(config.MODELS_DIR, "vocab.pkl")
    model_path = os.path.join(config.MODELS_DIR, f"{model_type}_best.pt")

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary not found at {vocab_path}. Train the model first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")

    vocab = Vocabulary.load(vocab_path)

    if model_type == "lstm":
        from src.lstm_model import CaptioningLSTM
        model = CaptioningLSTM(vocab_size=len(vocab))
    else:
        from src.transformer_model import CaptioningTransformer
        model = CaptioningTransformer(vocab_size=len(vocab))

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    image_tensor = load_image(image_path, device)

    if method == "beam":
        return beam_search(model, image_tensor, vocab)
    else:
        return model.caption(image_tensor, vocab)[0]
