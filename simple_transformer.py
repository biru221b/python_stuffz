import math
import torch

# simple_transformer.py
# A minimal transformer (encoder-decoder) implemented with PyTorch.
# Save as a single file and run with: python simple_transformer.py

import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


def generate_square_subsequent_mask(sz):
    # mask for decoder self-attention (prevent attending to future positions)
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask  # True means masked


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        # x: (batch, seq, d_model) -> (batch, heads, seq, d_head)
        b, seq, _ = x.size()
        return x.view(b, seq, self.num_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x):
        # x: (batch, heads, seq, d_head) -> (batch, seq, d_model)
        b, heads, seq, d_head = x.size()
        return x.transpose(1, 2).contiguous().view(b, seq, heads * d_head)

    def forward(self, q, k, v, mask=None):
        # q,k,v: (batch, seq, d_model)
        q = self._split_heads(self.q_lin(q))
        k = self._split_heads(self.k_lin(k))
        v = self._split_heads(self.v_lin(v))
        # q,k,v: (batch, heads, seq, d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            # mask: (seq_q, seq_k) or broadcastable; True means masked
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)  # (batch, heads, seq_q, d_head)
        context = self._merge_heads(context)
        return self.out_lin(context)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_out = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        self_attn = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(self_attn))
        cross = self.cross_attn(x, memory, memory, mask=memory_mask)
        x = self.norm2(x + self.dropout(cross))
        ff = self.ff(x)
        x = self.norm3(x + self.dropout(ff))
        return x


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_enc_layers=2,
        n_dec_layers=2,
        num_heads=4,
        d_ff=256,
        max_len=512,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(n_enc_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(n_dec_layers)]
        )

        self.out = nn.Linear(d_model, vocab_size)

    def encode(self, src, src_mask=None):
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask=src_mask)
        return x

    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # src: (batch, src_len), tgt: (batch, tgt_len)
        memory = self.encode(src, src_mask=src_mask)
        out = self.decode(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        logits = self.out(out)  # (batch, tgt_len, vocab)
        return logits


if __name__ == "__main__":
    # Quick smoke test with random data
    batch_size = 2
    src_len = 7
    tgt_len = 5
    vocab_size = 100

    device = torch.device("cpu")
    model = SimpleTransformer(vocab_size=vocab_size).to(device)

    src = torch.randint(0, vocab_size, (batch_size, src_len), device=device)
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len), device=device)

    # Create masks (optional). True = masked position.
    src_mask = None  # e.g. key padding mask shape (src_len, src_len)
    tgt_mask = generate_square_subsequent_mask(tgt_len)  # (tgt_len, tgt_len)
    logits = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

    print("Logits shape:", logits.shape)  # (batch, tgt_len, vocab_size)