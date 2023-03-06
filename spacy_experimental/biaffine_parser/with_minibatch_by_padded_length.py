from typing import Callable, Generic, List, Optional, Sized, Tuple, TypeVar, cast
from dataclasses import dataclass
from spacy.training.batchers import minibatch_by_padded_size
from thinc.api import Model


OutT = TypeVar("OutT")
SizedInT = TypeVar("SizedInT", bound=Sized)


@dataclass
class ItemIndex(Generic[SizedInT]):
    value: SizedInT
    idx: int

    def __len__(self):
        return len(self.value)


def with_minibatch_by_padded_length(
    inner: Model[List[SizedInT], List[OutT]], *, max_items=4096
) -> Model[List[SizedInT], List[OutT]]:
    """Batch the inputs sorted by length and with a maximum number of
    padded batch items."""
    return Model(
        "with_minibatch_by_length",
        with_minibatch_by_length_forward,
        init=with_minibatch_by_length_init,
        attrs={"max_items": max_items},
        layers=[inner],
    )


def with_minibatch_by_length_init(
    model: Model[List[SizedInT], List[OutT]], X: Optional[SizedInT] = None, Y=None
) -> None:
    # Pass X through as-is. Downstream models don't need the batching
    # for proper initialization.
    model.layers[0].initialize(X=X, Y=Y)


def with_minibatch_by_length_forward(
    model: Model[List[SizedInT], List[OutT]],
    X: List[SizedInT],
    is_train: bool,
) -> Tuple[List[OutT], Callable[[List[OutT]], List[SizedInT]]]:
    inner: Model[List[SizedInT], List[OutT]] = model.layers[0]
    max_items: int = model.attrs["max_items"]

    # Enumerate to keep track of the original order.
    splits_sorted = sorted(
        (ItemIndex(idx=idx, value=split) for idx, split in enumerate(X)),
        key=lambda i: len(i),
    )

    backprops = []
    Y: List[Optional[OutT]] = [None] * len(X)
    for batch in minibatch_by_padded_size(splits_sorted, max_items):
        X_batch = [split.value for split in batch]
        Y_batch, backprop_batch = inner(X_batch, is_train)
        backprops.append(backprop_batch)

        # Place in outputs.
        offsets_batch = [split.idx for split in batch]
        for split_offset, Y_split in zip(offsets_batch, Y_batch):
            Y[split_offset] = Y_split

    assert not any(y is None for y in Y)

    def backprop(dY: List[OutT]) -> List[SizedInT]:
        dX: List[Optional[SizedInT]] = [None] * len(X)
        for idx, batch in enumerate(minibatch_by_padded_size(splits_sorted, max_items)):
            dY_batch = [dY[split.idx] for split in batch]
            for split, dX_split in zip(batch, backprops[idx](dY_batch)):
                dX[split.idx] = dX_split

        assert not any(dx is None for dx in dX)

        return cast(List[SizedInT], dX)

    return cast(List[OutT], Y), backprop
