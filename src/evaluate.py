"""
src/evaluate.py
BLEU score evaluation on the test set.
"""

import torch
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from tqdm import tqdm
import config


def compute_bleu(model, loader, vocab, device=config.DEVICE):
    """
    Run model on test loader and compute BLEU-1 through BLEU-4.

    Returns:
        dict with keys bleu1, bleu2, bleu3, bleu4
    """
    model.eval()
    model.to(device)

    references = []   # list of list-of-reference-lists (per image)
    hypotheses = []   # list of token lists

    smoother = SmoothingFunction().method1

    with torch.no_grad():
        for images, captions, lengths in tqdm(loader, desc="Evaluating BLEU"):
            images   = images.to(device)
            captions = captions.to(device)

            # Generate captions (greedy)
            pred_strings = model.caption(images, vocab)

            for i in range(images.size(0)):
                # Reference: original caption (strip specials)
                ref_ids  = captions[i].tolist()
                ref_toks = vocab.decode(ref_ids).split()
                references.append([ref_toks])     # NLTK expects list of refs

                # Hypothesis
                hyp_toks = pred_strings[i].split()
                hypotheses.append(hyp_toks)

    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(1/3,)*3 + (0,))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25,)*4)

    results = dict(bleu1=bleu1, bleu2=bleu2, bleu3=bleu3, bleu4=bleu4)
    print("\n── BLEU Scores ──────────────────")
    for k, v in results.items():
        print(f"  {k.upper()}: {v:.4f}")
    print("─────────────────────────────────\n")
    return results
