"""
src/transformer_model.py
CNN + Transformer image captioning model.

Architecture:
  Image → ResNet50 → linear projection → Transformer Encoder memory
  Caption tokens → Positional Encoding → Transformer Decoder → vocab logits
"""

import math
import torch
import torch.nn as nn
import config
from src.feature_extractor import CNNEncoder


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    """
    Transformer-based language decoder conditioned on image memory.

    Args:
        vocab_size : vocabulary size
        d_model    : embedding / model dimension
        nhead      : number of attention heads
        num_layers : number of transformer decoder layers
        dropout    : dropout rate
    """

    def __init__(self,
                 vocab_size : int   = config.VOCAB_SIZE,
                 d_model    : int   = config.D_MODEL,
                 nhead      : int   = config.NHEAD,
                 num_layers : int   = config.NUM_DECODER_LAYERS,
                 dropout    : float = config.TRANSFORMER_DROPOUT):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, dropout)

        dec_layer = nn.TransformerDecoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = config.DIM_FEEDFORWARD,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,   # Pre-LN for stability
        )
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    @staticmethod
    def _causal_mask(size: int, device) -> torch.Tensor:
        """Upper-triangular mask to prevent attending to future tokens."""
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def forward(self,
                memory  : torch.Tensor,   # (B, 1, d_model) — image features
                captions: torch.Tensor,   # (B, T) — token indices
                pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Returns:
            logits : (B, T, vocab_size)
        """
        T   = captions.size(1)
        tgt = self.pos_enc(self.embed(captions) * math.sqrt(self.d_model))  # (B,T,d)
        causal = self._causal_mask(T, captions.device)

        out    = self.transformer_decoder(
            tgt                 = tgt,
            memory              = memory,
            tgt_mask            = causal,
            tgt_key_padding_mask= pad_mask,
        )  # (B, T, d_model)
        return self.fc(out)   # (B, T, vocab_size)


class CaptioningTransformer(nn.Module):
    """Full model: CNN Encoder + Transformer Decoder."""

    def __init__(self, vocab_size: int = config.VOCAB_SIZE):
        super().__init__()
        self.encoder   = CNNEncoder(feature_dim=config.D_MODEL)
        self.decoder   = TransformerDecoder(vocab_size=vocab_size)

    def forward(self, images: torch.Tensor,
                captions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images   : (B, 3, 224, 224)
            captions : (B, T)  with <START> ... <END>

        Returns:
            logits   : (B, T-1, vocab_size)  predicting tokens 1..T
        """
        memory  = self.encoder(images).unsqueeze(1)  # (B, 1, d_model)
        tgt_in  = captions[:, :-1]                   # drop last token
        pad_mask = (tgt_in == 0)                     # True where <PAD>
        return self.decoder(memory, tgt_in, pad_mask)

    def caption(self, images: torch.Tensor, vocab,
                max_len: int = config.MAX_DECODE_LEN) -> list:
        """Greedy auto-regressive decoding."""
        self.eval()
        B      = images.size(0)
        device = images.device
        with torch.no_grad():
            memory  = self.encoder(images).unsqueeze(1)  # (B,1,d)
            tokens  = torch.full((B, 1), vocab.stoi[config.START_TOKEN],
                                 device=device, dtype=torch.long)
            finished = [False] * B

            for _ in range(max_len):
                logits = self.decoder(memory, tokens)   # (B, t, V)
                next_t = logits[:, -1, :].argmax(-1)    # (B,)
                tokens = torch.cat([tokens, next_t.unsqueeze(1)], dim=1)
                for i in range(B):
                    if next_t[i].item() == vocab.stoi[config.END_TOKEN]:
                        finished[i] = True
                if all(finished):
                    break

        captions = []
        for i in range(B):
            ids = tokens[i, 1:].tolist()   # skip <START>
            captions.append(vocab.decode(ids))
        return captions
