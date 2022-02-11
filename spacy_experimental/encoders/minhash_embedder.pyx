# cython: infer_types=True, binding=True

from typing import List

from libc.stdint cimport uint32_t, uint64_t, UINT64_MAX
from murmurhash.mrmr cimport hash128_x64
import numpy
cimport numpy as np
from spacy.tokens import Doc
from thinc.api import Model
from thinc.types import Floats2d
from tokenizers import Tokenizer


def MinHashEmbed(n_hashes: int, token_features: int, vocab: List[str]) -> Model:
    pass


def extract_ngrams(piece: str, n: int) -> List[str]:
    if len(piece) < n:
        return [piece]

    ngrams = []
    while len(piece) >= n:
        ngrams.append(piece[:n])
        piece = piece[1:]

    return ngrams

def embed_tokens(table: VocabTable, tokenizer: Tokenizer, doc: Doc, n_features: int):
    tokens = [token.text for token in doc]
    encoding = tokenizer.encode(tokens, is_pretokenized=True, add_special_tokens=False)
    bloom = numpy.zeros((len(tokens), n_features), dtype="f")
    for i in range(len(tokens)):
        start, end = encoding.word_to_tokens(i)
        # Safe to cast to int since n_features is smaller than the maximum.
        hash_indices = (table.lookup(encoding.tokens[start:end]) % n_features).astype("i")
        bloom[i] = numpy.bincount(hash_indices, minlength=n_features)

    return bloom



cdef class VocabTable:
    def __init__(self, vocab: List[str], *, uint32_t hash_seed=42, size_t n_hashes=64):
        min_hashes = numpy.full((len(vocab), n_hashes), UINT64_MAX, dtype='u8')

        for i, piece in enumerate(vocab):
            # Continuation piece
            # TODO: use a function rather than hardcoding.
            if piece.startswith('##'):
                ngrams = [piece]
            else:
                ngrams = extract_ngrams(piece, 3)

            minhash_multiple(ngrams, hash_seed, n_hashes, min_hashes[i])

        self.hashes = min_hashes
        self.vocab = {piece: idx for idx, piece in enumerate(vocab) }

    def lookup(self, list pieces, out=None):
        indices = [self.vocab[piece] for piece in pieces]
        return self.hashes[indices].min(axis=0, out=out)

    @property
    def vocab_hashes(self):
        """Returns a copy of the vocab hashes"""
        return numpy.copy(self.hashes)

cdef void minhash_multiple(list ngrams, uint32_t hash_seed, size_t n_hashes, uint64_t [::1] out):
    cdef size_t i
    cdef char * c_chars
    cdef uint64_t hash
    cdef uint64_t[2] hash_parts

    for ngram in ngrams:
        chars = ngram.encode("utf8")
        c_chars = chars
        hash128_x64(c_chars, len(chars), hash_seed, &hash_parts)

        for i in range(n_hashes):
            hash = hash_parts[0] + i * hash_parts[1]
            if hash < out[i]:
                out[i] = hash
