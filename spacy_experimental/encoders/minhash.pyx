# cython: infer_types=True, binding=True

from libc.stdint cimport uint32_t, uint64_t, UINT64_MAX
from murmurhash.mrmr cimport hash128_x64
cimport numpy as np

cpdef void minhash_multiple(list ngrams, uint32_t hash_seed, size_t n_hashes, uint64_t [::1] out):
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
