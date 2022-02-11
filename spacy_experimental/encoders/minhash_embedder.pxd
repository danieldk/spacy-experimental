cimport numpy as np

cdef class VocabTable:
    cdef np.ndarray hashes
    cdef dict vocab
