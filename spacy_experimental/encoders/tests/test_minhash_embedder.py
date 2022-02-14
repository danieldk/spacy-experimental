import numpy as np
from numpy.testing._private.utils import assert_allclose
from spacy.vocab import Vocab
from spacy.tokens import Doc
import tokenizers

from spacy_experimental.encoders.minhash_embedder import VocabTable, embed_tokens


def sample_doc():
    words = ["hello", "world", "worldlings"]
    spaces = [True, False, False]
    vocab = Vocab()
    return Doc(vocab, words=words, spaces=spaces)


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
    out = np.zeros(3, dtype="u8")
    np.testing.assert_allclose(
        table.lookup(["hello", "##ing"]),
        [646525935662823063, 4453848894996444099, 1098919852625566271],
    )

    table.lookup(["hello", "##ing"], out=out),
    np.testing.assert_allclose(
        out,
        [646525935662823063, 4453848894996444099, 1098919852625566271],
    )

    np.testing.assert_allclose(
        table.lookup(["world", "##ing"]),
        [3403186037323527541, 3691063079667146669, 47369931100659919],
    )
    table.lookup(["world", "##ing"], out=out),
    np.testing.assert_allclose(
        out,
        [3403186037323527541, 3691063079667146669, 47369931100659919],
    )


def test_embeding_tokens():
    # tokenizer = tokenizers.models.WordPiece(
    #    {"hello": 0, "world": 1, "##ing": 2, "[UNK]": 3}, unk_token="[UNK]"
    # )
    tokenizer = tokenizers.Tokenizer.from_pretrained("bert-base-uncased")
    vocab = [tokenizer.id_to_token(i) for i in range(tokenizer.get_vocab_size())]
    table = VocabTable(vocab, hash_seed=42, n_hashes=3)
    embeds = embed_tokens(table, tokenizer, sample_doc(), n_features=3)
    assert_allclose(
        embeds,
        [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [1.0, 1.0, 1.0]],
    )

    embeds = embed_tokens(table, tokenizer, sample_doc(), n_features=3, nW=1)
    assert_allclose(
        embeds,
        [
            [0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 1.0],
            [2.0, 0.0, 1.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ],
    )
