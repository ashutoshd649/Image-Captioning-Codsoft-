"""
tests/test_all.py
Unit tests for vocabulary, dataset, models, and inference.
Run with: python -m pytest tests/ -v
"""

import os
import sys
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─── Vocabulary Tests ─────────────────────────────────────────────────────────

class TestVocabulary:
    def setup_method(self):
        from src.vocabulary import Vocabulary
        self.vocab = Vocabulary()
        captions = [
            "a dog runs in the park",
            "two children play on the swings",
            "a cat sits on a mat",
            "a man rides a bicycle",
            "a woman walks her dog",
        ]
        self.vocab.build(captions, max_size=100, min_freq=1)

    def test_special_tokens_present(self):
        assert config.PAD_TOKEN   in self.vocab.stoi
        assert config.START_TOKEN in self.vocab.stoi
        assert config.END_TOKEN   in self.vocab.stoi
        assert config.UNK_TOKEN   in self.vocab.stoi

    def test_special_token_indices(self):
        assert self.vocab.stoi[config.PAD_TOKEN]   == 0
        assert self.vocab.stoi[config.START_TOKEN] == 1
        assert self.vocab.stoi[config.END_TOKEN]   == 2
        assert self.vocab.stoi[config.UNK_TOKEN]   == 3

    def test_encode_decode_roundtrip(self):
        caption = "a dog runs in the park"
        encoded = self.vocab.encode(caption)
        decoded = self.vocab.decode(encoded)
        assert decoded == caption

    def test_unknown_word(self):
        encoded = self.vocab.encode("supercalifragilistic")
        assert encoded == [self.vocab.stoi[config.UNK_TOKEN]]

    def test_vocab_size(self):
        assert len(self.vocab) > 4   # at least special tokens + some words

    def test_itos_stoi_consistent(self):
        for idx, word in self.vocab.itos.items():
            assert self.vocab.stoi[word] == idx

    def test_save_load(self, tmp_path):
        from src.vocabulary import Vocabulary
        save_path = str(tmp_path / "vocab.pkl")
        self.vocab.save(save_path)
        loaded = Vocabulary.load(save_path)
        assert loaded.stoi == self.vocab.stoi
        assert len(loaded) == len(self.vocab)


# ─── Feature Extractor Tests ──────────────────────────────────────────────────

class TestCNNEncoder:
    def test_resnet50_output_shape(self):
        from src.feature_extractor import CNNEncoder
        encoder = CNNEncoder(model_name="resnet50", feature_dim=256)
        encoder.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = encoder(x)
        assert out.shape == (2, 256), f"Expected (2, 256), got {out.shape}"

    def test_vgg16_output_shape(self):
        from src.feature_extractor import CNNEncoder
        encoder = CNNEncoder(model_name="vgg16", feature_dim=256)
        encoder.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = encoder(x)
        assert out.shape == (2, 256)

    def test_frozen_backbone(self):
        from src.feature_extractor import CNNEncoder
        encoder = CNNEncoder(fine_tune=False)
        for p in encoder.backbone.parameters():
            assert not p.requires_grad

    def test_fine_tune_toggle(self):
        from src.feature_extractor import CNNEncoder
        encoder = CNNEncoder(fine_tune=False)
        encoder.fine_tune(True)
        for p in encoder.backbone.parameters():
            assert p.requires_grad

    def test_invalid_model_raises(self):
        from src.feature_extractor import CNNEncoder
        with pytest.raises(ValueError):
            CNNEncoder(model_name="alexnet")


# ─── LSTM Model Tests ─────────────────────────────────────────────────────────

class TestLSTMModel:
    def setup_method(self):
        from src.lstm_model import CaptioningLSTM
        from src.vocabulary import Vocabulary
        self.vocab = Vocabulary()
        self.vocab.build(["a dog runs fast", "cat on mat", "man walks bike"], min_freq=1)
        self.model = CaptioningLSTM(vocab_size=len(self.vocab))
        self.model.eval()
        self.B = 2
        self.T = 10

    def test_forward_shape(self):
        images   = torch.randn(self.B, 3, 224, 224)
        captions = torch.randint(1, len(self.vocab), (self.B, self.T))
        with torch.no_grad():
            logits = self.model(images, captions)
        assert logits.shape == (self.B, self.T - 1, len(self.vocab))

    def test_greedy_caption_returns_strings(self):
        images = torch.randn(self.B, 3, 224, 224)
        caps = self.model.caption(images, self.vocab)
        assert len(caps) == self.B
        for c in caps:
            assert isinstance(c, str)

    def test_encoder_output_shape(self):
        images = torch.randn(self.B, 3, 224, 224)
        with torch.no_grad():
            feats = self.model.encoder(images)
        assert feats.shape == (self.B, config.EMBED_DIM)


# ─── Transformer Model Tests ──────────────────────────────────────────────────

class TestTransformerModel:
    def setup_method(self):
        from src.transformer_model import CaptioningTransformer
        from src.vocabulary import Vocabulary
        self.vocab = Vocabulary()
        self.vocab.build(["a dog runs fast", "cat on mat", "man walks bike"], min_freq=1)
        self.model = CaptioningTransformer(vocab_size=len(self.vocab))
        self.model.eval()
        self.B = 2
        self.T = 10

    def test_forward_shape(self):
        images   = torch.randn(self.B, 3, 224, 224)
        captions = torch.randint(1, len(self.vocab), (self.B, self.T))
        with torch.no_grad():
            logits = self.model(images, captions)
        assert logits.shape == (self.B, self.T - 1, len(self.vocab))

    def test_greedy_caption_returns_strings(self):
        images = torch.randn(self.B, 3, 224, 224)
        caps = self.model.caption(images, self.vocab)
        assert len(caps) == self.B
        for c in caps:
            assert isinstance(c, str)

    def test_causal_mask(self):
        from src.transformer_model import TransformerDecoder
        mask = TransformerDecoder._causal_mask(5, "cpu")
        assert mask.shape == (5, 5)
        # upper triangle should be True (masked)
        assert mask[0, 1] == True
        assert mask[1, 0] == False


# ─── Positional Encoding Tests ────────────────────────────────────────────────

class TestPositionalEncoding:
    def test_output_shape(self):
        from src.transformer_model import PositionalEncoding
        pe = PositionalEncoding(d_model=64)
        x  = torch.randn(3, 20, 64)
        out = pe(x)
        assert out.shape == x.shape

    def test_different_positions_differ(self):
        from src.transformer_model import PositionalEncoding
        pe = PositionalEncoding(d_model=64, dropout=0.0)
        # Build PE matrix directly
        import math
        d_model, max_len = 64, 50
        enc = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        enc[:, 0::2] = torch.sin(pos * div)
        enc[:, 1::2] = torch.cos(pos * div)
        # Position 0 and position 1 should differ
        assert not torch.allclose(enc[0], enc[1])


# ─── Dataset Tests ────────────────────────────────────────────────────────────

class TestDataset:
    def test_collate_fn_padding(self):
        from src.dataset import collate_fn
        img1 = torch.randn(3, 224, 224)
        img2 = torch.randn(3, 224, 224)
        cap1 = torch.tensor([1, 5, 7, 2])
        cap2 = torch.tensor([1, 3, 2])
        images, padded, lengths = collate_fn([(img1, cap1), (img2, cap2)])
        assert images.shape  == (2, 3, 224, 224)
        assert padded.shape  == (2, 4)           # padded to max length
        assert lengths.tolist() == [4, 3]
        assert padded[1, 3].item() == 0          # last token of shorter cap = PAD


# ─── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
