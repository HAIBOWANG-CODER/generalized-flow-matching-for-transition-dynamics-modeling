import torch
import torch.nn as nn
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

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        q = self.split_heads(self.wq(query), batch_size)
        k = self.split_heads(self.wk(key), batch_size)
        v = self.split_heads(self.wv(value), batch_size)

        output = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)  ####################

        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        output = self.dense(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()                       #########################
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.gelu(self.linear1(x))))


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
        x = self.layernorm1(x)              ########################
        attn_output = self.multihead_attention(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        x = x + attn_output

        x = self.layernorm2(x)              ########################
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output)
        x = x + ffn_output

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, max_len, dropout=0.1, input_dim=7):
        super(TransformerEncoder, self).__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x = self.positional_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x, mask)

        output = self.output_layer(x)
        return output.reshape(x.shape[0], -1)


class VelocityNet(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, max_len: int):
        super().__init__()
        self.model = TransformerEncoder(d_model, num_heads, d_ff, num_layers, max_len, input_dim=4)

    def forward(self, t, x, **kwargs):
        x = x.reshape(x.shape[0], -1, 3)
        t = t[:, None, None].repeat(1, x.size(1), 1)
        x = torch.cat([t, x], dim=-1)
        return self.model(x)

class SplineNet(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, max_len: int,
                 time_spline: bool = False, flatten_input_reshape_output=None):
        super().__init__()

        self.time_spline = time_spline
        self.mainnet = TransformerEncoder(d_model, num_heads, d_ff, num_layers, max_len, input_dim=(7 if time_spline else 6))

        self.flatten_input_reshape_output = flatten_input_reshape_output

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x0 = x0.reshape(x0.shape[0], -1, 3)
        x1 = x1.reshape(x1.shape[0], -1, 3)
        t = t[:, None, None].repeat(1, x0.size(1), 1)

        x = torch.cat([x0, x1], dim=-1)
        if self.time_spline:
            x = torch.cat([x, t], dim=-1)

        out = self.mainnet(x)

        if self.flatten_input_reshape_output is not None:
            out = out.view(out.shape[0], *self.flatten_input_reshape_output)
        return out


d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
max_len = 166
dropout = 0.1

def main():
    t = torch.rand(16)

    spline_model = SplineNet(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_len=max_len,
        time_spline=True,
        flatten_input_reshape_output=(166, 3)
    )

    velocity_model = VelocityNet(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_len=max_len
    )

    x0 = torch.randn(16, 498)
    x1 = torch.randn(16, 498)

    output_spline = spline_model(x0, x1, t)
    print("SplineNet 输出形状:", output_spline.shape)

    output_velocity = velocity_model(t, x0)
    print("VelocityNet 输出形状:", output_velocity.shape)

if __name__ == "__main__":
    main()
