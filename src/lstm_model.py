"""
src/lstm_model.py
CNN + LSTM image captioning model.

Architecture:
  Image → ResNet/VGG → linear projection → LSTM hidden state h_0
  At each step: [word_embedding; image_features] → LSTM → linear → vocab logits
"""

import torch
import torch.nn as nn
import config
from src.feature_extractor import CNNEncoder


class LSTMDecoder(nn.Module):
    """
    LSTM language model conditioned on image features.

    Args:
        vocab_size  : total vocabulary size
        embed_dim   : word embedding dimension
        hidden_dim  : LSTM hidden state size
        num_layers  : number of LSTM layers
        dropout     : dropout probability
    """

    def __init__(self,
                 vocab_size : int = config.VOCAB_SIZE,
                 embed_dim  : int = config.EMBED_DIM,
                 hidden_dim : int = config.HIDDEN_DIM,
                 num_layers : int = config.NUM_LAYERS,
                 dropout    : float = config.DROPOUT):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, embed_dim,
                                      padding_idx=0)  # idx 0 = <PAD>
        self.lstm      = nn.LSTM(embed_dim + embed_dim,  # word + image
                                 hidden_dim,
                                 num_layers  = num_layers,
                                 batch_first = True,
                                 dropout     = dropout if num_layers > 1 else 0.0)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc.weight,    -0.1, 0.1)
        nn.init.constant_(self.fc.bias, 0)

    def init_hidden(self, img_features: torch.Tensor):
        """Create initial (h_0, c_0) from image features."""
        B = img_features.size(0)
        h0 = img_features.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (L, B, H)
        # zero-pad if hidden_dim != embed_dim
        if h0.size(-1) < self.hidden_dim:
            pad = torch.zeros(self.num_layers, B,
                              self.hidden_dim - h0.size(-1),
                              device=h0.device)
            h0 = torch.cat([h0, pad], dim=-1)
        else:
            h0 = h0[..., :self.hidden_dim]
        c0 = torch.zeros_like(h0)
        return h0, c0

    def forward(self, img_features: torch.Tensor,
                captions: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forcing forward pass.

        Args:
            img_features : (B, embed_dim)  from CNNEncoder
            captions     : (B, T)          token indices, T includes <START>

        Returns:
            logits : (B, T, vocab_size)
        """
        B, T = captions.shape
        word_emb = self.dropout(self.embed(captions))          # (B, T, E)
        img_exp  = img_features.unsqueeze(1).expand(-1, T, -1) # (B, T, E)
        lstm_in  = torch.cat([word_emb, img_exp], dim=-1)      # (B, T, 2E)

        h0, c0 = self.init_hidden(img_features)
        out, _ = self.lstm(lstm_in, (h0, c0))                  # (B, T, H)
        logits  = self.fc(self.dropout(out))                   # (B, T, V)
        return logits


class CaptioningLSTM(nn.Module):
    """Full model: Encoder + LSTM Decoder."""

    def __init__(self, vocab_size: int = config.VOCAB_SIZE):
        super().__init__()
        self.encoder = CNNEncoder()
        self.decoder = LSTMDecoder(vocab_size=vocab_size)

    def forward(self, images: torch.Tensor,
                captions: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)         # (B, embed_dim)
        logits   = self.decoder(features, captions[:, :-1])  # shift right
        return logits

    def caption(self, images: torch.Tensor, vocab,
                max_len: int = config.MAX_DECODE_LEN) -> list:
        """Greedy decoding — returns list of caption strings."""
        self.eval()
        with torch.no_grad():
            features = self.encoder(images)
            B        = features.size(0)
            input_   = torch.full((B, 1),
                                  vocab.stoi[config.START_TOKEN],
                                  device=features.device, dtype=torch.long)
            h, c = self.decoder.init_hidden(features)
            captions = [[] for _ in range(B)]
            finished = [False] * B

            for _ in range(max_len):
                word_emb = self.decoder.embed(input_)              # (B,1,E)
                img_exp  = features.unsqueeze(1)                   # (B,1,E)
                lstm_in  = torch.cat([word_emb, img_exp], dim=-1)  # (B,1,2E)
                out, (h, c) = self.decoder.lstm(lstm_in, (h, c))
                logits   = self.decoder.fc(out.squeeze(1))         # (B, V)
                preds    = logits.argmax(dim=-1)                   # (B,)
                input_   = preds.unsqueeze(1)

                for i in range(B):
                    if not finished[i]:
                        tok = preds[i].item()
                        if tok == vocab.stoi[config.END_TOKEN]:
                            finished[i] = True
                        else:
                            captions[i].append(tok)
                if all(finished):
                    break

        return [vocab.decode(c) for c in captions]
