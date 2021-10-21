from spacy import registry
from spacy.ml import extract_spans
from spacy.tokens.doc import Doc
from thinc.layers.pytorchwrapper import PyTorchWrapper_v2
from thinc.shims.pytorch_grad_scaler import PyTorchGradScaler
import torch.nn as nn
from thinc.api import (
    Model,
    Padded,
    PyTorchWrapper,
    Ragged,
    chain,
    get_width,
    list2array,
    list2ragged,
    with_array,
    with_getitem,
    with_padded,
    xp2torch,
    torch2xp,
)
from thinc.types import ArgsKwargs, Array2d, Floats2d, Floats3d, Ints1d
from typing import Any, Optional, List, Tuple, cast

from .pytorch_biaffine_model import BiaffineModel as PyTorchBiaffineModel


@registry.architectures("BiaffineModel.v1")
def build_biaffine_model(
    tok2vec: Model[List[Doc], List[Floats2d]],
    nO: Optional[int] = None,
    *,
    hidden_size: int = 128,
    mixed_precision: bool = False,
    grad_scaler: Optional[PyTorchGradScaler] = None
):
    biaffine = Model(
        "biaffine-model",
        forward=biaffine_forward,
        init=biaffine_init,
        dims={"nI": None, "nO": nO},
        attrs={"hidden_size": hidden_size},
    )
    model = chain(
        with_getitem(0, chain(tok2vec, list2array())), biaffine
    )
    return model


def biaffine_init(model: Model, X=None, Y=None):
    if model.layers:
        return

    if X is not None and model.has_dim("nI") is None:
        model.set_dim("nI", get_width(X))
    if Y is not None and model.has_dim("nO") is None:
        model.set_dim("nO", get_width(Y))

    hidden_size = model.attrs["hidden_size"]

    model._layers = [
        PyTorchWrapper_v2(
            PyTorchBiaffineModel(
                model.get_dim("nI"),
                model.get_dim("nO"),
                hidden_size=hidden_size,
            ),
            convert_inputs=convert_inputs,
            convert_outputs=convert_outputs,
        )
    ]


def biaffine_forward(model: Model, X, is_train: bool):
    return model.layers[0](X, is_train)

def convert_inputs(model: Model, Xr_lenghts: Tuple[Ragged, Ints1d], is_train: bool = False):
    flatten = model.ops.flatten
    unflatten = model.ops.unflatten
    pad = model.ops.pad
    unpad = model.ops.unpad

    Xr, lengths = Xr_lenghts

    Xt = xp2torch(pad(unflatten(Xr, lengths)), requires_grad=True)
    Lt = xp2torch(lengths)

    def convert_from_torch_backward(d_inputs: ArgsKwargs) -> Tuple[Floats2d, Ints1d]:
        dX = cast(Floats3d, torch2xp(d_inputs.args[0]))
        return flatten(unpad(dX, list(lengths))), lengths

    output = ArgsKwargs(args=(Xt, Lt), kwargs={})

    return output, convert_from_torch_backward

def convert_outputs(model, inputs_outputs, is_train):
    flatten = model.ops.flatten
    unflatten = model.ops.unflatten
    pad = model.ops.pad
    unpad = model.ops.unpad

    (_, lengths), Ytorch = inputs_outputs

    def convert_for_torch_backward(dY: Floats2d) -> ArgsKwargs:
        dYtorch = xp2torch(pad(unflatten(dY, lengths)))
        return ArgsKwargs(args=(Ytorch,), kwargs={"grad_tensors": dYtorch})

    Y = cast(Floats3d, torch2xp(Ytorch))
    Y = flatten(unpad(Y, lengths))

    return Y, convert_for_torch_backward
