from typing import Callable, List

import numpy as np
from spacy import registry, tokens
from spacy.tokens import Doc
from thinc.api import Linear, Model, chain, get_ops, with_array
from thinc.types import Floats2d, Ints2d
from tokenizers import Tokenizer

from .minhash import minhash_multiple


class VocabTable:
    def __init__(self, vocab: List[str], *, hash_seed: int = 42, n_hashes: int = 64):
        min_hashes = np.full((len(vocab), n_hashes), np.iinfo("u8").max)

        for i, piece in enumerate(vocab):
            # Continuation piece
            # TODO: use a function rather than hardcoding.
            if piece.startswith("##"):
                ngrams = [piece]
            else:
                ngrams = extract_ngrams(piece, 3)

            minhash_multiple(ngrams, hash_seed, n_hashes, min_hashes[i])

        self.hashes = min_hashes
        self.vocab = {piece: idx for idx, piece in enumerate(vocab)}

    def lookup(self, pieces: List[str], out=None):
        indices = [self.vocab[piece] for piece in pieces]
        return self.hashes[indices].min(axis=0, out=out)

    @property
    def n_hashes(self):
        return self.hashes.shape[1]

    @property
    def vocab_hashes(self):

        """Returns a copy of the vocab hashes"""
        return np.copy(self.hashes)


def embed_tokens(
    table: VocabTable, tokenizer: Tokenizer, doc: Doc, n_features: int, nW: int = 0
):
    tokens = [token.text for token in doc]
    encoding = tokenizer.encode(tokens, is_pretokenized=True, add_special_tokens=False)
    hashes = np.zeros((len(tokens), table.n_hashes), dtype="u8")
    for i in range(len(tokens)):
        start, end = encoding.word_to_tokens(i)
        o = table.lookup(encoding.tokens[start:end], out=hashes[i])

    # Convert to indices into the counting Bloom filter
    ops = get_ops("cpu")
    # Safe to cast to int since n_features is smaller than the maximum.
    hash_indices = (hashes % n_features).astype("i")

    bloom = np.zeros((len(tokens), n_features), dtype="f")
    for i in range(len(tokens)):
        bloom[i] = ops.xp.bincount(hash_indices[i], minlength=n_features)

    if nW > 0:
        bloom = ops.seq2col(bloom, nW)

    return bloom

@registry.architectures("spacy-experimental.MinHashEmbed.v1")
def MinHashEmbed(tokenizer_name: str, n_hashes: int, n_features: int, window_size: int, bottleneck: int, width: int, seed: int=0):
    return chain(
        MinHashFeatures(tokenizer_name, n_hashes, n_features, window_size),
        with_array(Linear(nO=bottleneck)),
        with_array(Linear(nO=width))
    )


def MinHashFeatures(
    tokenizer_name: str, n_hashes: int, n_features: int, window_size: int
) -> Model:
    tokenizer = Tokenizer.from_pretrained(tokenizer_name)
    vocab = [tokenizer.id_to_token(i) for i in range(tokenizer.get_vocab_size())]
    table = VocabTable(vocab, n_hashes=n_hashes)

    attrs = {
        "n_features": n_features,
        "tokenizer": tokenizer,
        "vocab_table": table,
        "window_size": window_size,
    }

    return Model("minhash_features", forward=forward, attrs=attrs)


def forward(model: Model[List[Doc], List[Ints2d]], docs: List[Doc], is_train: bool):
    tokenizer = model.attrs["tokenizer"]
    table = model.attrs["vocab_table"]
    nW = model.attrs["window_size"]
    nF = model.attrs["n_features"]

    embeds = []
    for doc in docs:
        embed = model.ops.asarray2f(embed_tokens(table, tokenizer, doc, nF, nW))
        embeds.append(embed)

    backprop: Callable[[List[Ints2d]], List] = lambda d_features: []
    return embeds, backprop


def extract_ngrams(piece: str, n: int) -> List[str]:
    if len(piece) < n:
        return [piece]

    ngrams = []
    while len(piece) >= n:
        ngrams.append(piece[:n])
        piece = piece[1:]

    return ngrams
