import numpy
from thinc.api import get_array_module
from thinc.types import Ints1d


def lens2offsets(lens: Ints1d) -> Ints1d:
    xp = get_array_module(lens)
    starts_ends = xp.empty(len(lens) + 1, dtype="i")
    starts_ends[0] = 0
    lens.cumsum(out=starts_ends[1:])
    return starts_ends[:-1]
