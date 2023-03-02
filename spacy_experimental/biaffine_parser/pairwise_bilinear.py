from dataclasses import dataclass
from typing import List, Optional, Tuple, cast
from cupy._creation.from_data import numpy

from spacy import registry
from spacy.tokens.doc import Doc
from spacy.training.batchers import minibatch_by_padded_size
from thinc.api import Model, Ops, chain, get_width, list2array, to_numpy, torch2xp
from thinc.api import with_getitem, xp2torch
from thinc.shims.pytorch_grad_scaler import PyTorchGradScaler
from thinc.types import ArgsKwargs, Floats2d, Floats3d, Floats4d, Ints1d

from ._util import lens2offsets

# Ensure that the spacy-experimental package can register entry points without
# Torch installed.
PyTorchPairwiseBilinearModel: Optional[type]
try:
    from .pytorch_pairwise_bilinear import (
        PairwiseBilinearModel as PyTorchPairwiseBilinearModel,
    )
except ImportError:
    PyTorchPairwiseBilinearModel = None


def build_pairwise_bilinear(
    tok2vec: Model[List[Doc], List[Floats2d]],
    nO=None,
    *,
    dropout: float = 0.1,
    hidden_width: int = 128,
    max_items: int = 4096,
    mixed_precision: bool = False,
    grad_scaler: Optional[PyTorchGradScaler] = None
) -> Model[Tuple[List[Doc], Ints1d], Floats2d]:
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
        with_getitem(0, tok2vec),
        with_padded_max_items(
            with_pad_sequence_unpad_bilinear(pairwise_bilinear), max_items
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
):
    X, L = X_lenghts

    Xt = xp2torch(X, requires_grad=is_train)
    Lt = xp2torch(L)

    def convert_from_torch_backward(d_inputs: ArgsKwargs) -> Tuple[Floats3d, Ints1d]:
        dX = cast(Floats3d, torch2xp(d_inputs.args[0]))
        return dX

    output = ArgsKwargs(args=(Xt, Lt), kwargs={})

    return output, convert_from_torch_backward


def convert_outputs(model, inputs_outputs, is_train):
    flatten = model.ops.flatten
    unflatten = model.ops.unflatten
    pad = model.ops.pad
    unpad = model.ops.unpad

    (_, lengths), Y_t = inputs_outputs

    def convert_for_torch_backward(dY: Tuple[Floats2d, Floats3d]) -> ArgsKwargs:
        dY_t = xp2torch(dY)
        return ArgsKwargs(
            args=([Y_t],),
            kwargs={"grad_tensors": [dY_t]},
        )

    Y = cast(Floats4d, torch2xp(Y_t))

    return Y, convert_for_torch_backward


def with_padded_max_items(inner: Model, max_items: int) -> Model:
    return Model(
        "with_padded_max_items",
        with_padded_max_items_forward,
        init=with_padded_max_items_init,
        attrs={"max_items": max_items},
        layers=[inner],
    )


def with_padded_max_items_init(model: Model, X=None, Y=None) -> None:
    # TODO: pass through X
    model.layers[0].initialize(Y=Y)


@dataclass
class Split:
    doc_id: int
    doc_offset: int
    split_offset: int
    array: Floats2d

    def __len__(self):
        return self.array.shape[0]


def minibatch_by_length(
    ops: Ops,
    inner: Model,
    X: List[Floats2d],
    splits: List[Split],
    max_items: int,
    is_train: bool,
):
    # Sort by split length
    splits_sorted = sorted(splits, key=lambda i: i.array.shape[0])

    backprops = []
    Y: List[Optional[Floats2d]] = [None] * len(splits)
    for batch in minibatch_by_padded_size(splits_sorted, max_items):
        X_batch = [split.array for split in batch]
        lens_batch = [X_split.shape[0] for X_split in X_batch]
        offsets_batch = [split.split_offset for split in batch]

        Y_batch, backprop = inner((X_batch, lens_batch), is_train)
        backprops.append(backprop)

        # Place in outputs.
        for split_offset, Y_split in zip(offsets_batch, Y_batch):
            Y[split_offset] = Y_split.reshape((-1,))

    def backprop(dY):
        dX_docs = [ops.alloc2f(*X_doc.shape, zeros=False) for X_doc in X]
        for idx, batch in enumerate(minibatch_by_padded_size(splits_sorted, max_items)):
            dY_batch = [dY[split.split_offset] for split in batch]
            dX_batch = backprops[idx](dY_batch)

            for split, dX_split in zip(batch, dX_batch):
                length = dX_split.shape[0]
                dX_docs[split.doc_id][
                    split.doc_offset : split.doc_offset + length
                ] = dX_split

        return dX_docs

    return Y, backprop


def with_pad_sequence_unpad_bilinear(inner: Model[List[Floats2d], List[Floats2d]]):
    """This layer is similar to with_padded, however it unpads
    correctly for layers that go from sequences to matrices."""
    return Model(
        "with_pad_seq_unpad_bilinear",
        with_pad_seq_unpad_bilinear_forward,
        init=with_pad_seq_unpad_bilinear_init,
        layers=[inner],
    )


def with_pad_seq_unpad_bilinear_init(model: Model, X=None, Y=None):
    inner = model.layers[0]
    if X is not None:
        X_seqs, lens = X
        inner.initialize((model.ops.pad(X_seqs), lens), Y)
    else:
        inner.initialize(X, Y)


def with_pad_seq_unpad_bilinear_forward(model: Model, X_lens, is_train):
    inner = model.layers[0]
    X, lens = X_lens

    X_padded = model.ops.pad(X)
    Y_padded, backprop_inner = inner((X_padded, model.ops.asarray1i(lens)), is_train)
    Y = unpad_matrix(Y_padded, lens)

    def backprop(dY):
        dY_padded = pad_matrix(model.ops, dY)
        dX_padded = backprop_inner(dY_padded)
        dX = model.ops.unpad(dX_padded, lens)
        return dX

    return Y, backprop


def with_padded_max_items_forward(
    model: Model, X_lens: Tuple[List[Floats2d], List[List[int]]], is_train: bool
):
    inner = model.layers[0]
    max_items: int = model.attrs["max_items"]

    X, lens = X_lens

    splits = []
    for doc_id, (X_doc, lens_docs) in enumerate(zip(X, lens)):
        split_offsets = lens2offsets(lens_docs)
        for split_offset, split_len in zip(split_offsets, lens_docs):
            splits.append(
                Split(
                    doc_id=doc_id,
                    doc_offset=split_offset,
                    split_offset=len(splits),
                    array=X_doc[split_offset : split_offset + split_len],
                )
            )

    Y, backprop_minibatch = minibatch_by_length(
        model.ops, inner, X, splits, max_items, is_train
    )

    def backprop(dY):
        dY_splits = []
        for split_len in [len for doc_lens in lens for len in doc_lens]:
            dY_splits.append(dY[: split_len * split_len].reshape(split_len, split_len))
            dY = dY[split_len * split_len :]

        assert dY.size == 0

        return backprop_minibatch(dY_splits), lens

    return model.ops.flatten(Y), backprop


def unpad_matrix(padded: Floats3d, lengths: List[int]) -> List[Floats2d]:
    """The reverse/backward operation of the `pad` function: transform an
    array back into a list of arrays, each with their original length.

    Different from Thinc, because it operates on a matrix with padding
    in two dimensions.
    """
    output = []
    for i, length in enumerate(lengths):
        output.append(padded[i, :length, :length])
    return cast(List[Floats2d], output)


def pad_matrix(ops: Ops, seqs: List[Floats2d], round_to=1) -> Floats3d:
    """Perform padding on a list of arrays so that they each have the same
    length, by taking the maximum dimension across each axis. This only
    works on non-empty sequences with the same `ndim` and `dtype`.

    Different from Thinc, because it operates on a matrix with padding
    in two dimensions.
    """
    if not seqs:
        raise ValueError("Cannot pad empty sequence")
    if len(set(seq.ndim for seq in seqs)) != 1:
        raise ValueError("Cannot pad sequences with different ndims")
    if len(set(seq.dtype for seq in seqs)) != 1:
        raise ValueError("Cannot pad sequences with different dtypes")
    if any(len(seq.shape) != 2 or seq.shape[0] != seq.shape[1] for seq in seqs):
        raise ValueError("Cannot pad non-square matrices")
    # Find the maximum dimension along each axis. That's what we'll pad to.
    length = max(seq.shape[0] for seq in seqs)
    # Round the length to nearest bucket -- helps on GPU, to make similar
    # array sizes.
    length = (length + (round_to - 1)) // round_to * round_to
    final_shape = (len(seqs), length, length)
    output = ops.alloc3f(*final_shape)
    for i, arr in enumerate(seqs):
        # It's difficult to convince this that the dtypes will match.
        output[i, : arr.shape[0], : arr.shape[1]] = arr  # type: ignore[assignment, call-overload]
    return output
