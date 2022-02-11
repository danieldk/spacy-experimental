import numpy as np

from spacy_experimental.encoders.minhash_embedder import VocabTable


def test_create_vocab_table():
    pieces = ["hello", "world", "##ing"]
    table = VocabTable(pieces, hash_seed=42, n_hashes=3)
    np.testing.assert_allclose(
        table.vocab_hashes,
        [
            [646525935662823063, 4453848894996444099, 1098919852625566271],
            [3403186037323527541, 3691063079667146669, 47369931100659919],
            [14659342254517029925, 12708937444018081563, 10758532633519133201],
        ],
    )


def test_vocab_table_lookup():
    pieces = ["hello", "world", "##ing"]
    table = VocabTable(pieces, hash_seed=42, n_hashes=3)
    np.testing.assert_allclose(
        table.lookup(["hello", "##ing"]),
        [646525935662823063, 4453848894996444099, 1098919852625566271],
    )
    np.testing.assert_allclose(
        table.lookup(["world", "##ing"]),
        [3403186037323527541, 3691063079667146669, 47369931100659919],
    )
