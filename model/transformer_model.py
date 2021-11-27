import math

from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from model.positional_encoding import PositionalEncoding


class TransformerModel(nn.Module):
    def __init__(
        self,
        n_token: int,
        d_model: int = 200,  # embedding dim
        n_head: int = 2,  # #heads in nn.MultiheadAttention
        d_hid: int = 200,  # dim ff net in nn.TransformerEncoder
        n_layers: int = 2,  # #layers in TransformerEncoder
        dropout: float = 0.2,
        device: int = 0,
    ):
        super().__init__()

        self.n_token = n_token
        self.d_model = d_model

        self.encoder = nn.Embedding(n_token, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model,
            n_head,
            d_hid,
            dropout,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.decoder = nn.Linear(d_model, n_token)

        self.to(device)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor):

        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)

        return output.view(-1, self.n_token)
