from typing import Iterable, Iterator, TypeVar
import torch
from recurrent.mytypes import PARAMETER
from torch.utils import _pytree as pytree


def rnnSplitParameters(
    n_h: int, n_in: int, n_out: int, parameters: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    w_rec, w_out = torch.split(parameters, [n_h * (n_h + n_in + 1), n_out * (n_h + 1)])
    return w_rec, w_out


def jacobian(outs, inps) -> torch.Tensor:
    outs = torch.atleast_1d(outs)
    I_N = torch.eye(outs.size(0))

    def get_vjp(v):
        return torch.autograd.grad(outs, inps, v, create_graph=True, retain_graph=True)[
            0
        ]

    return torch.vmap(get_vjp)(I_N)


def tree_stack(trees: Iterable[pytree.PyTree]) -> pytree.PyTree:
    return pytree.tree_map(lambda *v: torch.stack(v), *trees)


def tree_unstack(tree: pytree.PyTree) -> Iterator[pytree.PyTree]:
    leaves, treedef = pytree.tree_flatten(tree)
    return (treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True))
