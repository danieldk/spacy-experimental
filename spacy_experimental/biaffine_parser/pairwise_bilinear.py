from typing import Callable, List, Optional, Tuple, TypeVar, Union
from typing import cast
from spacy import registry
from spacy.tokens.doc import Doc
from thinc.api import Model, chain, get_width, torch2xp
from thinc.api import with_getitem, xp2torch
from thinc.shims.pytorch_grad_scaler import PyTorchGradScaler
from thinc.types import (
    ArgsKwargs,
    Floats1d,
    Floats2d,
    Floats3d,
    Floats4d,
    Ints1d,
)

from ._util import lens2offsets
from .with_minibatch_by_padded_size import with_minibatch_by_padded_size
from .with_pad_seq_unpad_matrix import with_pad_seq_unpad_matrix

# Ensure that the spacy-experimental package can register entry points without
# Torch installed.
PyTorchPairwiseBilinearModel: Optional[type]
try:
    from .pytorch_pairwise_bilinear import (
        PairwiseBilinearModel as PyTorchPairwiseBilinearModel,
    )
except ImportError:
    PyTorchPairwiseBilinearModel = None


InT = TypeVar("InT")
OutT = TypeVar("OutT")


def build_pairwise_bilinear(
    tok2vec: Model[List[Doc], List[Floats2d]],
    nO=None,
    *,
    dropout: float = 0.1,
    hidden_width: int = 128,
    max_items: int = 4096,
    mixed_precision: bool = False,
    grad_scaler: Optional[PyTorchGradScaler] = None
) -> Model[Tuple[List[Doc], List[Ints1d]], Floats1d]:
    if PyTorchPairwiseBilinearModel is None:
        raise ImportError(
            "PairwiseBiLinear layer requires PyTorch: pip install thinc[torch]"
        )

    nI = None
    if tok2vec.has_dim("nO") is True:
        nI = tok2vec.get_dim("nO")

    pairwise_bilinear: Model[Tuple[Floats2d, Ints1d], Floats2d] = Model(
        "pairwise_bilinear",
        forward=pairwise_bilinear_forward,
        init=pairwise_bilinear_init,
        dims={"nI": nI, "nO": nO},
        attrs={
            # We currently do not update dropout when dropout_rate is
            # changed, since we cannot access the underlying model.
            "dropout_rate": dropout,
            "hidden_width": hidden_width,
            "mixed_precision": mixed_precision,
            "grad_scaler": grad_scaler,
        },
    )

    model = chain(
        cast(
            Model[Tuple[List[Doc], List[Ints1d]], Tuple[List[Floats2d], List[Ints1d]]],
            with_getitem(0, tok2vec),
        ),
        with_splits(
            with_minibatch_by_padded_size(
                with_pad_seq_unpad_matrix(pairwise_bilinear), size=max_items
            )
        ),
    )
    model.set_ref("pairwise_bilinear", pairwise_bilinear)

    return model


def pairwise_bilinear_init(model: Model, X=None, Y=None):
    if PyTorchPairwiseBilinearModel is None:
        raise ImportError(
            "PairwiseBiLinear layer requires PyTorch: pip install thinc[torch]"
        )

    if model.layers:
        return

    if X is not None and model.has_dim("nI") is None:
        model.set_dim("nI", get_width(X))
    if Y is not None and model.has_dim("nO") is None:
        model.set_dim("nO", get_width(Y))

    hidden_width = model.attrs["hidden_width"]
    mixed_precision = model.attrs["mixed_precision"]
    grad_scaler = model.attrs["grad_scaler"]

    PyTorchWrapper = registry.get("layers", "PyTorchWrapper.v2")
    model._layers = [
        PyTorchWrapper(
            PyTorchPairwiseBilinearModel(
                model.get_dim("nI"),
                model.get_dim("nO"),
                dropout=model.attrs["dropout_rate"],
                hidden_width=hidden_width,
            ),
            convert_inputs=convert_inputs,
            convert_outputs=convert_outputs,
            mixed_precision=mixed_precision,
            grad_scaler=grad_scaler,
        )
    ]


def pairwise_bilinear_forward(model: Model, X, is_train: bool):
    return model.layers[0](X, is_train)


def convert_inputs(
    model: Model, X_lenghts: Tuple[Floats2d, Ints1d], is_train: bool = False
) -> Tuple[ArgsKwargs, Callable[[ArgsKwargs], Floats3d]]:
    X, L = X_lenghts

    Xt = xp2torch(X, requires_grad=is_train)
    Lt = xp2torch(L)

    def convert_from_torch_backward(d_inputs: ArgsKwargs) -> Floats3d:
        dX = cast(Floats3d, torch2xp(d_inputs.args[0]))
        return dX

    output = ArgsKwargs(args=(Xt, Lt), kwargs={})

    return output, convert_from_torch_backward


def convert_outputs(
    model, inputs_outputs, is_train: bool
) -> Tuple[
    Union[Floats3d, Floats4d], Callable[[Union[Floats3d, Floats4d]], ArgsKwargs]
]:
    (_, lengths), Y_t = inputs_outputs

    def convert_for_torch_backward(dY: Union[Floats3d, Floats4d]) -> ArgsKwargs:
        dY_t = xp2torch(dY)
        return ArgsKwargs(
            args=([Y_t],),
            kwargs={"grad_tensors": [dY_t]},
        )

    Y = cast(Union[Floats3d, Floats4d], torch2xp(Y_t))

    return Y, convert_for_torch_backward


def with_splits(
    inner: Model[List[Floats2d], List[Floats2d]]
) -> Model[Tuple[List[Floats2d], List[Ints1d]], Floats1d]:
    return Model(
        "with_splits",
        with_splits_forward,
        init=with_splits_init,
        layers=[inner],
    )


def with_splits_init(model: Model, X=None, Y=None) -> None:
    # TODO: pass through X
    model.layers[0].initialize(Y=Y)


def with_splits_forward(
    model: Model[Tuple[List[Floats2d], List[Ints1d]], Floats1d],
    X_lens: Tuple[List[Floats2d], List[Ints1d]],
    is_train: bool,
) -> Tuple[Floats1d, Callable[[Floats1d], Tuple[List[Floats2d], List[Ints1d]]]]:
    inner = model.layers[0]

    X, lens = X_lens

    splits = []
    split_locs = []
    for doc_id, (X_doc, lens_docs) in enumerate(zip(X, lens)):
        split_offsets = lens2offsets(lens_docs)
        for split_offset, split_len in zip(split_offsets, lens_docs):
            splits.append(X_doc[split_offset : split_offset + split_len])
            split_locs.append((doc_id, split_offset))

    Y, backprop_inner = inner(splits, is_train)

    def backprop(dY: Floats1d) -> Tuple[List[Floats2d], List[Ints1d]]:
        dY_splits = []
        for split_len in [len for doc_lens in lens for len in doc_lens]:
            dY_splits.append(
                dY[: split_len * split_len].reshape((split_len, split_len))
            )
            dY = dY[split_len * split_len :]

        assert dY.size == 0

        dX_splits = backprop_inner(dY_splits)

        dX_docs = [model.ops.alloc2f(*X_doc.shape, zeros=False) for X_doc in X]
        for (doc_id, doc_offset), dX_split in zip(split_locs, dX_splits):
            length = dX_split.shape[0]
            dX_docs[doc_id][doc_offset : doc_offset + length] = dX_split

        return dX_docs, lens

    return model.ops.flatten([Ys.reshape(-1) for Ys in Y]), backprop
