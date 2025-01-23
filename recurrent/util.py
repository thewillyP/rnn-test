from typing import Iterable, Iterator, TypeVar
import torch
from recurrent.mytypes import *
from torch.utils import _pytree as pytree
from recurrent.parameters import (
    RnnParameter,
)
from torch.utils._pytree import PyTree


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


def zeroedInfluenceTensor(out: int, param: PyTree):
    def update(x: torch.Tensor):
        n = torch.numel(x)
        return torch.zeros((out, n))

    return pytree.tree_map(update, param)


def zeroedParam(param: PyTree):
    return pytree.tree_map(lambda x: torch.zeros_like(x), param)


def fmapPytree(f, tree: PyTree):
    return pytree.tree_map(lambda x: f(x), tree)


def pytreeRepeatBatch(batch: int, tree: PyTree):
    return pytree.tree_map(
        lambda x: torch.repeat_interleave(x.unsqueeze(0), batch, dim=0), tree
    )


def uoroBInit(param: RnnParameter):
    return Gradient[RnnParameter](
        RnnParameter(
            w_rec=PARAMETER(torch.randn_like(param.w_rec)),
            w_out=PARAMETER(torch.zeros_like(param.w_out)),
        )
    )


def pytreeNumel(tree: PyTree):
    leafs, _ = pytree.tree_flatten(tree)
    return sum((torch.numel(x) for x in leafs))
