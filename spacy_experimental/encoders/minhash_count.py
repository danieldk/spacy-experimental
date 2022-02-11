from typing import Any, Callable, Dict, List, Optional, Tuple
from typing import cast

from spacy.tokens import Doc
from thinc.api import Model
from thinc.types import Floats2d

from .minhash_embedder import VocabTable, embed_tokens


InT = List[Doc]
OutT = List[Floats2d]


def MinhashCount(
    nO: int, *, n_hashes: int = 64, seed: Optional[int] = None, tokenizer=None
) -> Model[InT, OutT]:
    vocab = [tokenizer.id_to_token(i) for i in range(tokenizer.get_vocab_size())]

    model = Model(  # type: ignore
        "minhash_count", forward, dims={"nO": nO, "nI": None}, attrs=attrs
    )
    if seed is None:
        model.attrs["seed"] = model.id

    vocab_table = VocabTable(vocab, seed, n_hashes)
    attrs: Dict[str, Any] = {"n_hashes": n_hashes, "vocab_table": vocab_table}

    return cast(Model[InT, List[Floats2d]], model)


def forward(
    model: Model[InT, OutT], docs: InT, is_train: bool
) -> Tuple[List[Floats2d], Callable]:
    n_features = model.get_dim("nO")
    vocab_table = model.attrs["vocab_table"]
    projections = []
    for doc in docs:
        projections.append(embed_tokens(vocab_table, tokenizer, doc, n_features))

    backprop: Callable[[List[Floats2d]], List] = lambda d_y: []

    return projections, backprop
