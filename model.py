import torch
from torch import nn
import math

class Attention(nn.Module):
  def __init__(self, d_model = 512, num_heads = 8, dropout = 0.1):
    super().__init__()

    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = d_model // num_heads

    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)
    self.w_o = nn.Linear(d_model, d_model)

    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask = None): # mask is optional
    b, s, _ = x.size() # b = batch_size, s = seq_len
    # (bs, seq, num_heads, d_k) → (bs, num_heads, seq, d_k)
    q = self.w_q(x).view(b, s, self.num_heads, self.d_k).transpose(1,2)
    k = self.w_k(x).view(b, s, self.num_heads, self.d_k).transpose(1,2)
    v = self.w_v(x).view(b, s, self.num_heads, self.d_k).transpose(1,2)

    # scaling dividing by math.sqrt(self.d_k)
    attn_out = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
    if mask is not None:
      # mask changes pos 0 to -inf so that makes prob to 0 after softmax
      attn_out = attn_out.masked_fill(mask == 0, float('-inf'))
    attn_out = torch.softmax(attn_out, dim=-1)
    attn_out = self.dropout(attn_out)
    attn_out = torch.matmul(attn_out, v)
    attn_out = attn_out.transpose(1,2).reshape(b, s, self.d_model)
    attn_out = self.w_o(attn_out)

    return attn_out

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model = 512, num_heads = 8, dropout = 0.1):
    super().__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = d_model // num_heads

    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)
    self.w_o = nn.Linear(d_model, d_model)

    self.dropout = nn.Dropout(dropout)

  def forward(self, q, k, v, mask = None):
    b, q_len, _ = q.size() # b = batch_size, s = seq_len
    b, kv_len, _ = k.size()
    # (bs, seq, num_heads, d_k) → (bs, num_heads, seq, d_k)
    q = self.w_q(q).view(b, q_len, self.num_heads, self.d_k).transpose(1,2)
    k = self.w_k(k).view(b, kv_len, self.num_heads, self.d_k).transpose(1,2)
    v = self.w_v(v).view(b, kv_len, self.num_heads, self.d_k).transpose(1,2)

    # scaling dividing by math.sqrt(self.d_k)
    attn_out = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
    if mask is not None:
      # mask changes pos 0 to -inf so that makes prob to 0 after softmax
      attn_out = attn_out.masked_fill(mask == 0, float('-inf'))
    attn_out = torch.softmax(attn_out, dim=-1)
    attn_out = self.dropout(attn_out)
    attn_out = torch.matmul(attn_out, v)
    attn_out = attn_out.transpose(1,2).reshape(b, q_len, self.d_model)
    attn_out = self.w_o(attn_out)

    return attn_out

def look_ahead_mask_(size):
    mask = torch.tril(torch.ones(size, size))

    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)

class Encoder(nn.Module):
  def __init__(self, d_model = 512, num_heads = 8, dropout = 0.1):
    super().__init__()

    self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
    self.dropout1 = nn.Dropout(dropout)
    self.layer_norm1 = nn.LayerNorm(d_model)

    self.ffn = FeedForward(d_model)
    self.dropout2 = nn.Dropout(dropout)
    self.layer_norm2 = nn.LayerNorm(d_model)

  def forward(self, x, mask = None):
    attention_out = self.self_attention(x, x, x, mask)
    x = x + self.dropout1(attention_out)
    x = self.layer_norm1(x)

    ffn_out = self.ffn(x)
    x = x + self.dropout2(ffn_out)
    x = self.layer_norm2(x)

    return x

class Decoder(nn.Module):
  def __init__(self, d_model = 512, num_heads = 8, dropout = 0.1):
    super().__init__()

    self.self_attention = MultiHeadAttention(d_model, num_heads, dropout) #Masked MHA
    self.dropout1 = nn.Dropout(dropout)
    self.layer_norm1 = nn.LayerNorm(d_model)

    self.enc_dec_attention = MultiHeadAttention(d_model, num_heads, dropout) #Masked MHA
    self.dropout2 = nn.Dropout(dropout)
    self.layer_norm2 = nn.LayerNorm(d_model)

    self.ffn = FeedForward(d_model)
    self.dropout3 = nn.Dropout(dropout)
    self.layer_norm3 = nn.LayerNorm(d_model)

  def forward(self, x, enc_output, look_ahead_mask_ = None, mask = None):
    attention_out = self.self_attention(x, x, x, look_ahead_mask_)
    x = x + self.dropout1(attention_out)
    x = self.layer_norm1(x)

    attention_out = self.enc_dec_attention(x, enc_output, enc_output, mask)
    x = x + self.dropout2(attention_out)
    x = self.layer_norm2(x)

    ffn_out = self.ffn(x)
    x = x + self.dropout3(ffn_out)
    x = self.layer_norm3(x)

    return x

class EncoderStack(nn.Module):
    def __init__(self, num_layers = 6, d_model = 512, num_heads = 8, dropout = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            Encoder(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class DecoderStack(nn.Module):
    def __init__(self, num_layers = 6, d_model = 512, num_heads = 8, dropout = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            Decoder(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, enc_output, look_ahead_mask_=None, mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, look_ahead_mask_, mask)
        return x


class FeedForward(nn.Module):
  def __init__(self, d_model = 512, num_heads = 8, d_ff = 2048, dropout = 0.1):
    super().__init__()

    self.linear1 = nn.Linear(d_model, d_ff)
    self.relu1 = nn.ReLU()
    self.linear2 = nn.Linear(d_ff, d_model)

  def forward(self, x):
    x = self.linear1(x)
    x = self.relu1(x)
    x = self.linear2(x)

    return x

class Embedding(nn.Module):
  def __init__(self, vocab_size, d_model = 512): # vocab_size is the total number of words/tokens
    super().__init__()
    self.d_model = d_model
    self.emb = nn.Embedding(vocab_size, d_model)

  def forward(self, x):
    return self.emb(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
  def __init__(self, d_model = 512, max_len = 5000, dropout=0.1): # vocab_size is the total number of words/tokens
    super().__init__()
    self.d_model = d_model
    self.dropout = nn.Dropout(dropout)

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    dim = torch.arange(0, d_model, 2)
    pe[:, 0::2] = torch.sin(position / (10_000 ** (dim / self.d_model)))
    pe[:, 1::2] = torch.cos(position / (10_000 ** (dim / self.d_model)))

    pe = pe.unsqueeze(0)  # (1, max_len, d_model)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:, : x.size(1)].detach()
    return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()

        self.src_embedding = Embedding(src_vocab_size, d_model)
        self.tgt_embedding = Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        self.encoder = EncoderStack(num_layers, d_model, num_heads, dropout)
        self.decoder = DecoderStack(num_layers, d_model, num_heads, dropout)

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.src_embedding(src)
        src = self.pos_encoding(src)

        tgt = self.tgt_embedding(tgt)
        tgt = self.pos_encoding(tgt)

        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)

        output = self.fc_out(dec_output)
        return output