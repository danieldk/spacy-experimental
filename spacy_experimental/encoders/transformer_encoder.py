import torch

from spacy import registry
from spacy.ml.models.tok2vec import MultiHashEmbed
from torch import nn
from thinc.types import Floats2d, Floats3d, Ints1d, Ints2d, Padded, ArgsKwargs
from thinc.util import xp2torch, torch2xp
from typing import List, Tuple, cast
from thinc.api import Model, PyTorchWrapper, with_padded, chain

from torch import Tensor

from torch.nn import TransformerEncoder, TransformerEncoderLayer


# Same as for RNN, but passes lenghts as input
def convert_transformer_inputs(model: Model, Xp: Padded, is_train: bool):
    size_at_t = Xp.size_at_t
    lengths = Xp.lengths
    indices = Xp.indices

    def convert_from_torch_backward(d_inputs: ArgsKwargs) -> Padded:
        dX = torch2xp(d_inputs.args[0])
        return Padded(dX, size_at_t, lengths, indices)  # type: ignore

    X = xp2torch(Xp.data, requires_grad=True)
    L = xp2torch(lengths, requires_grad=False)
    output = ArgsKwargs(args=(X, L), kwargs={})
    return output, convert_from_torch_backward


# Same as for RNN, but with a single output
def convert_transformer_outputs(model: Model, inputs_outputs: Tuple, is_train):
    Xp, Ytorch = inputs_outputs

    def convert_for_torch_backward(dYp: Padded) -> ArgsKwargs:
        dYtorch = xp2torch(dYp.data, requires_grad=True)
        return ArgsKwargs(args=(Ytorch,), kwargs={"grad_tensors": dYtorch})

    Y = cast(Floats3d, torch2xp(Ytorch))
    Yp = Padded(Y, Xp.size_at_t, Xp.lengths, Xp.indices)
    return Yp, convert_for_torch_backward


def length_to_mask(
        length: Ints1d
) -> Ints2d:
    """
    length: batch.
    return B x max_len.
    """
    max_len = length.max().item()
    mask = torch.arange(
        max_len,
        device=length.device,
        dtype=length.dtype
    ).expand(len(length), max_len) > length.unsqueeze(1)
    return mask


# https://theaisummer.com/positional-embeddings/
class AbsPosEnc(nn.Module):
    """
    Learned absolute positional embeddings.
    """
    def __init__(
            self,
            dim: int,
            dropout: float = 0.1,
            max_len: int = 512
    ):
        super().__init__()
        # Positional Embedding matrix
        self.abs_pos_emb = nn.Parameter(torch.randn(max_len, dim))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor,  len x batch x dim
        """
        pos_emb = self.abs_pos_emb[:x.size(0)]
        pos_emb = pos_emb.unsqueeze(1).repeat(1, x.size(1), 1)
        x = x + pos_emb
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        max_len: int
    ):
        super().__init__()
        # Learned absolute position encodings
        self.pos_embedding = AbsPosEnc(input_dim, dropout)
        # Single transformer encoder layer
        encoder_layers = TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
        )
        # Stack of transformer encoder layers
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

    def forward(
        self,
        X: Tensor,
        lengths: Tensor
    ) -> Tensor:
        """
        Args:
            X: Tensor, len x batch x dim
            mask: Tensor, batch x len
        """
        mask = length_to_mask(lengths)
        X = self.pos_embedding(X)
        output = self.transformer_encoder(
            X,
            src_key_padding_mask=mask
        )
        return output


@registry.architectures("spacy-experimental.PyTorchTransformerEncoder.v1")
def PyTorchTransformerEncoder(
    width: int = 768,
    hidden_dim: int = 768,
    n_heads: int = 6,
    depth: int = 6,
    dropout: float = 0.2,
    max_len: int = 512
) -> Model[List[Floats2d], List[Floats2d]]:
    pytorch_transformer = TransformerModel(
        width,
        hidden_dim,
        n_heads,
        depth,
        dropout,
        max_len
    )
    transformer_encoder = PyTorchWrapper(
        pytorch_transformer,
        convert_inputs=convert_transformer_inputs,
        convert_outputs=convert_transformer_outputs
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
    embedder = MultiHashEmbed(
        width,
        attrs,
        rows,
        include_static_vectors
    )
    encoder = PyTorchTransformerEncoder(width=width)
    return chain(embedder, encoder)


if __name__ == "__main__":
    import spacy

    nlp = spacy.blank("en")
    text1 = 'hello i am a doc'
    text2 = 'i am a lovely teapot man'
    docs = [nlp(text1), nlp(text2)]
    transformer_tok2vec = create_default_model()
    transformer_tok2vec.initialize()
    feats, backprop = transformer_tok2vec(docs, False)
