"""
src/vocabulary.py
Build and manage the word vocabulary for captions.
"""

import re
from collections import Counter
import pickle
import os
import config


def tokenise(text: str):
    """Lowercase and split on non-alpha characters."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text.split()


class Vocabulary:
    """
    Maps words <-> integer indices.

    Usage:
        vocab = Vocabulary()
        vocab.build(list_of_captions)
        idx   = vocab.stoi["dog"]
        word  = vocab.itos[idx]
    """

    def __init__(self):
        self.stoi = {}   # string → index
        self.itos = {}   # index → string
        self.freq = Counter()

    def build(self, captions, max_size=config.VOCAB_SIZE, min_freq=config.MIN_WORD_FREQ):
        """Build vocabulary from a list of caption strings."""
        for cap in captions:
            self.freq.update(tokenise(cap))

        # Reserve special tokens at fixed indices
        special = [config.PAD_TOKEN, config.START_TOKEN, config.END_TOKEN, config.UNK_TOKEN]
        for i, tok in enumerate(special):
            self.stoi[tok] = i
            self.itos[i]   = tok

        # Add top-N frequent words
        common = [w for w, f in self.freq.most_common(max_size) if f >= min_freq]
        for word in common:
            if word not in self.stoi:
                idx = len(self.stoi)
                self.stoi[word] = idx
                self.itos[idx]  = word

        print(f"[Vocabulary] size = {len(self.stoi)} words")

    def encode(self, caption: str):
        """Convert caption string → list of integer indices."""
        unk = self.stoi[config.UNK_TOKEN]
        return [self.stoi.get(w, unk) for w in tokenise(caption)]

    def decode(self, indices):
        """Convert list of indices → caption string (strips special tokens)."""
        skip = {config.PAD_TOKEN, config.START_TOKEN, config.END_TOKEN}
        words = [self.itos.get(i, config.UNK_TOKEN) for i in indices]
        words = [w for w in words if w not in skip]
        return " ".join(words)

    def __len__(self):
        return len(self.stoi)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[Vocabulary] saved → {path}")

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            vocab = pickle.load(f)
        print(f"[Vocabulary] loaded ← {path}  (size={len(vocab)})")
        return vocab
