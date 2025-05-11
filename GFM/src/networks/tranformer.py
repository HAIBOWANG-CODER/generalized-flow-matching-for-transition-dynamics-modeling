import torch
import torch.nn as nn
import torch.nn.init as init
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=166):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Xavier/Glorot initialization for linear layers
        init.xavier_uniform_(self.wq.weight)
        init.xavier_uniform_(self.wk.weight)
        init.xavier_uniform_(self.wv.weight)
        init.xavier_uniform_(self.dense.weight)

        # Initialize biases to zero
        init.zeros_(self.wq.bias)
        init.zeros_(self.wk.bias)
        init.zeros_(self.wv.bias)
        init.zeros_(self.dense.bias)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        q = self.split_heads(self.wq(query), batch_size)
        k = self.split_heads(self.wk(key), batch_size)
        v = self.split_heads(self.wv(value), batch_size)

        output = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)

        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        output = self.dense(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Xavier/Glorot initialization for linear layers
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

        # Initialize biases to zero
        init.zeros_(self.linear1.bias)
        init.zeros_(self.linear2.bias)

    def forward(self, x):
        return self.linear2(self.gelu(self.dropout(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        _x = x
        x = self.layernorm1(x)
        attn_output = self.multihead_attention(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        x = x + attn_output

        _x = x
        x = self.layernorm2(x)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output)
        x = x + ffn_output

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, max_len, dropout=0.1, input_dim=7):
        super(TransformerEncoder, self).__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.layernorm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, 3)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Xavier/Glorot initialization for linear layers
        init.xavier_uniform_(self.input_proj.weight)
        init.xavier_uniform_(self.output_layer.weight)

        # Initialize biases to zero
        init.zeros_(self.input_proj.bias)
        init.zeros_(self.output_layer.bias)

        # Special initialization for output layer if needed
        init.normal_(self.output_layer.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x = self.positional_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x, mask)

        x = self.layernorm(x)
        output = self.output_layer(x)
        return output.reshape(x.shape[0], -1)