"""
MIT License

Copyright (c) 2020 Bodhisattwa Majumder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
"""

import math
import torch

from spacy import registry
from spacy.ml.models.tok2vec import MultiHashEmbed
from torch import nn
from thinc.types import Floats2d, Floats3d, Ints1d, Ints2d, Padded, ArgsKwargs
from thinc.util import xp2torch, torch2xp
from typing import List, Tuple, cast
from thinc.api import Model, with_padded, chain

# Use thinc.api >= 8.0.14
from thinc.layers.pytorchwrapper import PyTorchGradScaler, PyTorchWrapper_v2

from torch import Tensor
import torch.nn.functional as F
from torch.nn import LayerNorm, TransformerEncoder, TransformerEncoderLayer


# Same as for RNN, but passes lenghts as input
def convert_transformer_inputs(model: Model, Xp: Padded, is_train: bool):
    size_at_t = Xp.size_at_t
    lengths = Xp.lengths
    indices = Xp.indices

    def convert_from_torch_backward(d_inputs: ArgsKwargs) -> Padded:
        dX = torch2xp(d_inputs.args[0])
        return Padded(dX, size_at_t, lengths, indices)  # type: ignore

    X = xp2torch(Xp.data, requires_grad=is_train)
    L = xp2torch(lengths, requires_grad=False)
    output = ArgsKwargs(args=(X, L), kwargs={})
    return output, convert_from_torch_backward


# Same as for RNN, but with a single output
def convert_transformer_outputs(model: Model, inputs_outputs: Tuple, is_train):
    Xp, Ytorch = inputs_outputs

    def convert_for_torch_backward(dYp: Padded) -> ArgsKwargs:
        dYtorch = xp2torch(dYp.data, requires_grad=False)
        return ArgsKwargs(args=(Ytorch,), kwargs={"grad_tensors": dYtorch})

    Y = cast(Floats3d, torch2xp(Ytorch))
    Yp = Padded(Y, Xp.size_at_t, Xp.lengths, Xp.indices)
    return Yp, convert_for_torch_backward


def length_to_mask(length: Ints1d) -> Ints2d:
    """
    length: batch.
    return B x max_len.
    """
    max_len = length.max().item()
    mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(
        len(length), max_len
    ) >= length.unsqueeze(1)
    return mask


# https://theaisummer.com/positional-embeddings/
class AbsPosEnc(nn.Module):
    """
    Learned absolute positional embeddings.
    """

    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        # Positional Embedding matrix
        self.abs_pos_emb = nn.Parameter(torch.randn(max_len, dim))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor,  len x batch x dim
        """
        pos_emb = self.abs_pos_emb[: x.size(0)]
        pos_emb = pos_emb.unsqueeze(1).repeat(1, x.size(1), 1)
        x = x + pos_emb
        return self.dropout(x)


# Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class SinusoidalEncoding(nn.Module):
    def __init__(
        self, d_model: int, dropout: float = 0.1, max_len: int = 5000, normalize=True
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        if normalize == True:
            l2 = torch.norm(pe, dim=-1)
            pe /= l2.unsqueeze(-1)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# From https://github.com/majumderb/rezero/
# License https://github.com/majumderb/rezero/blob/master/LICENSE


class ReZeroEncoderLayer(nn.Module):
    """
    d_model: the number of expected features in the input (required).
    nhead: the number of heads in the multiheadattention models (required).
    dim_feedforward: the dimension of the feedforward network model (default=2048).
    dropout: the dropout value (default=0.1).
    activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.resweight = nn.Parameter(torch.Tensor([0.]))

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in PyTroch Transformer class.
        """
        # Self attention layer
        src2 = src
        src2 = self.self_attn(
            src2, src2, src2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src2 = src2[0]
        src2 = src2 * self.resweight
        src = src + self.dropout1(src2)

        # Pointiwse FF Layer
        src2 = src
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src2 = src2 * self.resweight
        src = src + self.dropout2(src2)
        return src


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        input_dropout: float,
        dropout: float,
        max_len: int,
        layer_norm_eps: float = 1e-5,
        rezero: bool = False,
        use_norm: bool = True
    ):
        super().__init__()
        # Learned absolute position encodings
        self.pos_embedding = AbsPosEnc(input_dim, input_dropout)
        # Single transformer encoder layer
        if rezero:
            encoder_layers = ReZeroEncoderLayer(
                d_model=input_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout
            )
        else:
            encoder_layers = TransformerEncoderLayer(
                d_model=input_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
            )
            # Stack of transformer encoder layers
        if use_norm:
            encoder_norm = LayerNorm(input_dim, eps=layer_norm_eps)
        else:
            encoder_norm = None
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, n_layers, encoder_norm
        )

    def forward(self, X: Tensor, lengths: Tensor) -> Tensor:
        """
        Args:
            X: Tensor, len x batch x dim
            mask: Tensor, batch x len
        """
        mask = length_to_mask(lengths)
        X = self.pos_embedding(X)
        output = self.transformer_encoder(X, src_key_padding_mask=mask)
        return output


@registry.architectures("spacy-experimental.PyTorchTransformerEncoder.v1")
def PyTorchTransformerEncoder(
    width: int = 768,
    hidden_dim: int = 768,
    n_heads: int = 6,
    depth: int = 6,
    input_dropout: float = 0.1,
    dropout: float = 0.2,
    max_len: int = 512,
    layer_norm_eps: float = 1e-5,
    rezero: bool = False,
    use_norm: bool = False,
    mixed_precision: bool = False,
    grad_scaler_config: dict = {},
) -> Model[List[Floats2d], List[Floats2d]]:
    pytorch_transformer = TransformerModel(
        width,
        hidden_dim,
        n_heads,
        depth,
        input_dropout,
        dropout,
        max_len,
        layer_norm_eps,
        rezero,
        use_norm
    )

    # Enable gradient scaling when mixed precision is enabled and gradient
    # scaling is not explicitly disabled in the configuration.
    if "enabled" not in grad_scaler_config:
        grad_scaler_config["enabled"] = mixed_precision

    transformer_encoder = PyTorchWrapper_v2(
        pytorch_transformer,
        convert_inputs=convert_transformer_inputs,
        convert_outputs=convert_transformer_outputs,
        mixed_precision=mixed_precision,
        grad_scaler=PyTorchGradScaler(**grad_scaler_config),
    )
    return with_padded(transformer_encoder)


def create_default_model():
    """
    Get default tok2vec with PyTorchTransformerEncoder.
    """
    attrs = ["NORM", "PREFIX", "SUFFIX", "LOWER"]
    rows = [5000, 2500, 2500, 2500]
    width = 300
    include_static_vectors = False
    embedder = MultiHashEmbed(width, attrs, rows, include_static_vectors)
    encoder = PyTorchTransformerEncoder(width=width, rezero=True)
    return chain(embedder, encoder)


if __name__ == "__main__":
    import spacy

    nlp = spacy.blank("en")
    text1 = "hello i am a doc"
    text2 = "i am a lovely teapot man"
    docs = [nlp(text1), nlp(text2)]
    transformer_tok2vec = create_default_model()
    transformer_tok2vec.initialize()
    feats, backprop = transformer_tok2vec(docs, False)
