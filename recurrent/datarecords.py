from dataclasses import dataclass
import torch
from torch.utils import _pytree as pytree

from recurrent.util import tree_unstack


@dataclass(frozen=True)
class Input2Output1:
    x1: torch.Tensor
    x2: torch.Tensor
    y: torch.Tensor

    def __iter__(self):
        return iter(tree_unstack(self))


def customdata_flatten(custom_data: Input2Output1):
    return (custom_data.x1, custom_data.x2, custom_data.y), None


def customdata_unflatten(children, _):
    return Input2Output1(x=children[0], y=children[1])


pytree.register_pytree_node(Input2Output1, customdata_flatten, customdata_unflatten)
