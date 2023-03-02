import numpy
from thinc.types import Ints1d


def lens2offsets(lens: Ints1d) -> Ints1d:
    starts_ends = numpy.empty(len(lens) + 1, dtype="i")
    starts_ends[0] = 0
    numpy.cumsum(lens, out=starts_ends[1:])
    return starts_ends[:-1]
