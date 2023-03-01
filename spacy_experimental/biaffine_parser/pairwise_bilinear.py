from dataclasses import dataclass
from typing import List, Optional, Tuple, cast

from spacy import registry
from spacy.tokens.doc import Doc
from spacy.training.batchers import minibatch_by_padded_size
from thinc.api import Model, Ops, chain, get_width, list2array, to_numpy, torch2xp
from thinc.api import with_getitem, xp2torch
from thinc.shims.pytorch_grad_scaler import PyTorchGradScaler
from thinc.types import ArgsKwargs, Floats2d, Floats3d, Floats4d, Ints1d

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

    model: Model[Tuple[List[Doc], Ints1d], Floats2d] = chain(
        cast(
            Model[Tuple[List[Doc], Ints1d], Tuple[List[Floats2d], Ints1d]],
            with_getitem(0, tok2vec),
        ),
        ## TODO: do not hardcode max items
        with_padded_max_items(pairwise_bilinear, 4096),
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
        return dX, L

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


def with_padded_max_items_forward(
    model: Model, X_lens: Tuple[List[Floats2d], Ints1d], is_train: bool
):
    inner = model.layers[0]
    max_items: int = model.attrs["max_items"]

    X, lens = X_lens

    lens = to_numpy(lens)

    # Extact all splits with offsets
    splits = []
    split_offset = 0
    for doc_id, X_doc in enumerate(X):
        doc_offset = 0
        while X_doc.size:
            splits.append(Split(doc_id=doc_id, doc_offset=doc_offset, split_offset=split_offset, array=X_doc[:lens[split_offset]]))
            doc_offset += int(lens[split_offset])
            X_doc = X_doc[lens[split_offset] :]
            split_offset += 1

    # Sort by split length
    splits.sort(key=lambda i: i.array.shape[0])

    backprops = []
    Y: List[Optional[Floats2d]] = [None] * len(splits)
    for batch in minibatch_by_padded_size(
        splits, max_items
    ):
        X_batch = [split.array for split in batch]
        lens_batch = [X_split.shape[0] for X_split in X_batch]
        offsets_batch = [split.split_offset for split in batch]

        X_padded = model.ops.pad(X_batch)
        Y_padded, backprop = inner((X_padded, model.ops.asarray1i(lens_batch)), is_train)
        backprops.append(backprop)
        Y_batch = unpad_matrix(Y_padded, lens_batch)

        # Place in outputs.
        for split_offset, Y_split in zip(offsets_batch, Y_batch):
            Y[split_offset] = Y_split.reshape((-1,))

    def backprop(dY):
        nonlocal backprops

        # TODO: dY is a 1D array of the concatenation of all matrixes for
        # all splits/docs in document order. reconstruct splits array. Then
        # go from there. Something like this. Lens still needs to be a copy,
        # since we mutate it above...
        dY_splits = []
        for split_len in lens:
            dY_splits.append(dY[:split_len*split_len].reshape(split_len, split_len))
            dY = dY[split_len*split_len:]

        assert dY.size == 0

        dX_docs = [model.ops.alloc2f(*X_doc.shape, zeros=False) for X_doc in X]
        for idx, batch in enumerate(minibatch_by_padded_size(
            splits, max_items
        )):
            dY_batch = [dY_splits[split.split_offset] for split in batch]
            dY_padded = pad_matrix(model.ops, dY_batch)
            dX_padded, L = backprops[idx](dY_padded)
            dX = model.ops.unpad(dX_padded, L)

            for split, dX_split in zip(batch, dX):
                length = dX_split.shape[0]
                dX_docs[split.doc_id][split.doc_offset: split.doc_offset + length] = dX_split


        return dX_docs, lens

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

def pad_matrix(  # noqa: F811
    ops: Ops, seqs: List[Floats2d], round_to=1
) -> Floats3d:
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
        output[i, : arr.shape[0], :arr.shape[1]] = arr  # type: ignore[assignment, call-overload]
    return output


@dataclass
class Split:
    doc_id: int
    doc_offset: int
    split_offset: int
    array: Floats2d

    def __len__(self):
        return self.array.shape[0]
