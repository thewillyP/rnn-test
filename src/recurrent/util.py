from dataclasses import dataclass
import jax
from recurrent.myfunc import compose2
from recurrent.mytypes import *
import equinox as eqx
import jax.numpy as jnp

from recurrent.parameters import RnnParameter


# https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
def tree_stack(trees):
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)


def tree_unstack(tree):
    leaves, treedef = jax.tree.flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]


def tree_unstack_lazy(tree):
    leaves, treedef = jax.tree.flatten(tree)
    return (treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True))


@eqx.filter_jit
def pytree_norm(tree):
    squared = jax.tree.map(lambda x: jnp.sum(x**2), tree)
    return jnp.sqrt(jax.tree.reduce(jnp.add, squared))


def prng_split(key: PRNG) -> tuple[PRNG, PRNG]:
    new_key, prng = jax.random.split(key)
    return PRNG(prng), PRNG(new_key)


# def zeroedInfluenceTensor(out: int, param: eqx.Module):
#     def update(x: jax.Array):
#         n = jnp.size(x)
#         return jnp.zeros((out, n))

#     return jax.tree.map(update, param)


def zeroedInfluenceTensor[T: CanDiff](out: int, param: T):
    n = jnp.size(compose2(endowVector, toVector)(param))
    return jnp.zeros((out, n))


def uoroBInit(key: jax.Array, param: RnnParameter):
    init = RnnParameter(
        w_rec=jax.random.normal(key, param.w_rec.shape),
        w_out=jnp.zeros_like(param.w_out),
    )
    return toVector(endowVector(init))


def pytreeNumel(tree: eqx.Module):
    leafs, _ = jax.tree.flatten(tree)
    return sum((jnp.size(x) for x in leafs))


def pytreeSumZero[T: eqx.Module](tree: T) -> T:
    return jax.tree.map(lambda x: jnp.zeros_like(x), tree)


@dataclass(frozen=True)
class SplitResult:
    reshaped: jax.Array
    leftover: jax.Array


# since this is python, there is no way to gaurantee the index assumptions hold, so whatever
def get_leading_dim_size(tree) -> int:
    leaves = jax.tree.leaves(tree)
    leading_dim_size = leaves[0].shape[0]
    return leading_dim_size


def split_and_reshape(arr: jax.Array, T: int):
    n_complete = (len(arr) // T) * T
    main_part = arr[:n_complete].reshape((-1, T) + arr.shape[1:])
    extra_part = arr[n_complete:]
    return SplitResult(main_part, extra_part)


def pytree_split[Tree: eqx.Module](tree: Tree, T: int) -> tuple[Tree, Tree]:
    reshaped_pytree_: Tree = jax.tree.map(lambda arr: split_and_reshape(arr, T), tree)
    reshaped_pytree = jax.tree.map(lambda x: x.reshaped, reshaped_pytree_)
    leftover_pytree = jax.tree.map(lambda x: x.leftover, reshaped_pytree_)
    return reshaped_pytree, leftover_pytree


# from typing import Iterable, Iterator, TypeVar
# import torch
# from recurrent.mytypes import *
# from torch.utils import _pytree as pytree
# from recurrent.parameters import (
#     RnnParameter,
# )
# from torch.utils._pytree import PyTree


# def rnnSplitParameters(
#     n_h: int, n_in: int, n_out: int, parameters: torch.Tensor
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     w_rec, w_out = torch.split(parameters, [n_h * (n_h + n_in + 1), n_out * (n_h + 1)])
#     return w_rec, w_out


# def jacobian(outs, inps) -> torch.Tensor:
#     outs = torch.atleast_1d(outs)
#     I_N = torch.eye(outs.size(0))

#     def get_vjp(v):
#         return torch.autograd.grad(outs, inps, v, create_graph=True, retain_graph=True)[
#             0
#         ]

#     return torch.vmap(get_vjp)(I_N)


# def tree_stack(trees: Iterable[pytree.PyTree]) -> pytree.PyTree:
#     return pytree.tree_map(lambda *v: torch.stack(v), *trees)


# def tree_unstack(tree: pytree.PyTree) -> Iterator[pytree.PyTree]:
#     leaves, treedef = pytree.tree_flatten(tree)
#     return (treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True))


# def zeroedInfluenceTensor(out: int, param: PyTree):
#     def update(x: torch.Tensor):
#         n = torch.numel(x)
#         return torch.zeros((out, n))

#     return pytree.tree_map(update, param)


# def zeroedParam(param: PyTree):
#     return pytree.tree_map(lambda x: torch.zeros_like(x), param)


# def fmapPytree(f, tree: PyTree):
#     return pytree.tree_map(lambda x: f(x), tree)


# def pytreeRepeatBatch(batch: int, tree: PyTree):
#     return pytree.tree_map(
#         lambda x: torch.repeat_interleave(x.unsqueeze(0), batch, dim=0), tree
#     )


# def uoroBInit(param: RnnParameter):
#     return Gradient[RnnParameter](
#         RnnParameter(
#             w_rec=PARAMETER(torch.randn_like(param.w_rec)),
#             w_out=PARAMETER(torch.zeros_like(param.w_out)),
#         )
#     )


# def pytreeNumel(tree: PyTree):
#     leafs, _ = pytree.tree_flatten(tree)
#     return sum((torch.numel(x) for x in leafs))
