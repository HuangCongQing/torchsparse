from itertools import repeat
from typing import List, Tuple, Union

import numpy as np
import torch

__all__ = ['make_ntuple']


def make_ntuple(x: Union[int, List[int], Tuple[int, ...], torch.Tensor],
                ndim: int) -> Tuple[int, ...]:
    if isinstance(x, int):
        x = tuple(repeat(x, 3))
    elif isinstance(x, list) or isinstance(x, tuple):
        x = tuple(np.array(x))
    elif isinstance(x, torch.Tensor):
        x = tuple(x.view(-1).cpu().numpy().tolist())

    x = tuple(float(d) for d in x)
    assert isinstance(x, tuple) and len(x) == ndim, x
    return x
