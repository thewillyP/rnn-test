# # from typing import NamedTuple
# # import torch
# # from torch.utils import _pytree as pytree


# # class MyPyTree[T](NamedTuple):
# #     x: T
# #     y: list[torch.Tensor]


# # # Register the class as a PyTree
# # def tree_flatten[T](obj: MyPyTree[T]):
# #     # Extract the tensors as a list and return auxiliary data
# #     return (obj.x, obj.y), None


# # def tree_unflatten[T](children, aux):
# #     # Reconstruct the object from its flattened representation
# #     return MyPyTree[T](*children)


# # pytree.register_pytree_node(MyPyTree, tree_flatten, tree_unflatten)


# # # Define a function to test differentiability
# # def func(pytree: list[MyPyTree[torch.Tensor]]):
# #     return (pytree[0].x ** 2).sum()


# # # Create an instance of MyPyTree with tensors
# # x = torch.tensor([1.0, 2.0])
# # p = MyPyTree[torch.Tensor](x, [torch.tensor([3.0, 4.0])])

# # # Compute the Jacobian using jacrev
# # print(p)
# # jacobian = torch.func.jacrev(func)([p])

# # # Print the Jacobian to verify
# # print(jacobian)

# # # j, tree = pytree.tree_flatten(jacobian)
# # tree = pytree.tree_structure([p])
# # zeroed = pytree.tree_unflatten((torch.tensor(0),) * tree.num_leaves, tree)


# # print(pytree.tree_map(lambda x, y: x + y, jacobian, zeroed))


# import torch
# from torch.func import vjp
# from collections import namedtuple
# from torch.utils import _pytree as pytree


# def pytree_norm(tree):
#     leaves, _ = pytree.tree_flatten(tree)
#     return torch.sqrt(sum(torch.sum(leaf**2) for leaf in leaves))


# # Define the namedtuple to serve as our pytree
# Pytree = namedtuple("Pytree", ["field1", "field2"])


# # Define a function that operates on the pytree and outputs a tensor of size n
# def my_function(pytree):
#     field1, field2 = pytree.field1, pytree.field2
#     # Perform some computation involving both fields
#     return field1 * torch.sin(field2) + field2 * torch.cos(field1)


# # Create an example pytree with two tensors
# n = 3
# example_pytree = Pytree(field1=torch.randn(n), field2=torch.randn(n))

# # Compute the VJP of my_function
# output, func_vjp = vjp(my_function, example_pytree)

# # Apply the VJP to a vector (seed for the adjoint computation)
# seed = torch.ones_like(output)
# vjp_result = func_vjp(seed)


# print("Function output:", output)
# print("VJP result:", vjp_result)
# print(pytree_norm(vjp_result))

# # jacobian = torch.func.jacrev(my_function)(example_pytree)
# # print(jacobian)


# # print(pytree.tree_map(lambda x: seed @ x, jacobian))


# import torch
# from torch.func import jvp


# # Define a function that computes a non-1D output based on a tensor input
# def func(x):
#     return torch.mm(x.T, x)  # Compute the outer product for a non-1D output


# # Define the primal input as a row tensor
# primal = torch.tensor([1.0, 2.0, 3.0])

# # print(torch.mm(primal.T, primal))

# # Define a tangent vector for the jvp computation
# tangent = torch.tensor([[0.1], [0.2], [0.3]])
# tangent = torch.flatten(tangent)

# # Compute the jvp (Jacobian-vector product)
# jvp_result = jvp(func, (primal,), (tangent,))


# print("Jacobian-vector product result:")
# print(jvp_result)


# from collections import namedtuple
# import torch

# # Triple nesting structure
# Inner = namedtuple("Inner", ["a", "b"])
# Middle = namedtuple("Middle", ["inner", "c"])
# Outer = namedtuple("Outer", ["middle", "d"])


# def transform_and_update(outer):
#     new_inner = Inner(outer.middle.inner.a + 1, outer.middle.inner.b * 2)
#     new_middle = Middle(new_inner, outer.middle.c**2)
#     updated = Outer(new_middle, outer.d + 1)
#     return (
#         updated.middle.inner.a + updated.middle.inner.b + updated.middle.c + updated.d,
#         updated,
#     )


# parametrized = lambda outer: transform_and_update(outer)
# pr = Outer(
#     Middle(Inner(torch.tensor(1.0), torch.tensor(2.0)), torch.tensor(3.0)),
#     torch.tensor(4.0),
# )
# result, env = torch.func.jacrev(parametrized, has_aux=True)(pr)

# print(parametrized(pr))
# print(env.middle.inner.a)
# print(result)


# import torch


# def recursive_operation(tensor, n):
#     if n == 0:
#         return tensor
#     else:
#         # Example operation: multiplying tensor by 2 (you can replace this with your desired operation)
#         result = tensor * 2
#         return recursive_operation(result, n - 1)


# # Example usage:
# input_tensor = torch.tensor([1.0, 2.0, 3.0])
# n = 5
# model = torch.compile(recursive_operation)
# output_tensor = model(input_tensor, n)
# print(output_tensor)


# import time
# import torch


# def create_recursive_closure(tensor, n):
#     def closure(tensor, remaining):
#         if remaining == 0:
#             return tensor
#         else:
#             # Example operation: multiplying tensor by 2 (you can replace this with your desired operation)
#             result = tensor * 2
#             return create_recursive_closure(result, remaining - 1)

#     return closure(tensor, n)


# # Example usage:
# input_tensor = torch.tensor([1.0, 2.0, 3.0])
# n = 5
# model = torch.compile(create_recursive_closure)
# start = time.time()
# output_tensor = model(input_tensor, n)
# print(time.time() - start)
# start = time.time()
# output_tensor = model(input_tensor, n)
# print(time.time() - start)
# start = time.time()
# output_tensor = model(input_tensor, n)
# print(time.time() - start)
# start = time.time()
# output_tensor = model(input_tensor, n)
# print(time.time() - start)
# start = time.time()
# output_tensor = model(input_tensor, n)
# print(time.time() - start)
# # print(output_tensor)


# import jax
# import jax.numpy as jnp
# from typing import Callable, NewType, Generator
# from donotation import do

# # Define a newtype for the ndarray
# ArrayWrapper = NewType("ArrayWrapper", jnp.ndarray)


# class Unit:
#     __slots__ = ()


# class Fold[D, E, S, A]:

#     __slots__ = ("func",)

#     def __init__(self, func: Callable[[D, E, S], tuple[A, S]]):
#         self.func = func

#     def __iter__(self) -> Generator[None, None, A]: ...  # type: ignore

#     def flat_map[B](self, func: Callable[[A], "Fold[D, E, S, B]"]):

#         def next(d: D, env: E, state: S) -> tuple[B, S]:
#             value, n_state = self.func(d, env, state)
#             return func(value).func(d, env, n_state)

#         return Fold(next)

#     def fmap[B](self, func: Callable[[A], B]):
#         def next(d: D, env: E, state: S) -> tuple[B, S]:
#             value, n_state = self.func(d, env, state)
#             return func(value), n_state

#         return Fold(next)

#     def then[B](self, m: "Fold[D, E, S, B]"):
#         return self.flat_map(lambda _: m)

#     def switch_dl[Pidgen](self, dl: D):
#         def next(_: Pidgen, env: E, state: S):
#             return self.func(dl, env, state)

#         return Fold(next)  # type: ignore


# def pure[D, E, S, A](value: A) -> Fold[D, E, S, A]:
#     return Fold(lambda _d, _e, state: (value, state))


# class Test:

#     def test(self, x):
#         @do()
#         def next():
#             print("recompiling")
#             y = yield from pure(x)
#             return pure(ArrayWrapper(jnp.sin(y) + jnp.cos(y)))

#         return next()


# # Function to wrap and operate on the array
# def wrapped_function(z: ArrayWrapper) -> ArrayWrapper:
#     o, _ = Test().test(z).func(None, None, None)
#     return o


# # JIT compile the function
# jit_wrapped_function = jax.jit(wrapped_function)

# # Example usage
# data = ArrayWrapper(jnp.array([0.0, jnp.pi / 4, jnp.pi / 2]))
# # f_safe = jit_wrapped_function.lower(data).compile()
# result = jit_wrapped_function(data)
# result = jit_wrapped_function(data)
# result = jit_wrapped_function(data)

# print("Input:", data)
# print("Result:", result)


# import jax
# import jax.numpy as jnp


# # Define a function that takes an ndarray and returns a float32
# def compute_sum_of_squares(x):
#     y = jnp.sum(x**2)
#     return jnp.float32(y)


# # Create an example ndarray
# x = jnp.array([1.0, 2.0, 3.0])

# # Autodiff to compute the gradient of the function with respect to x
# grad_fn = jax.grad(compute_sum_of_squares)
# grad = grad_fn(x)

# print("Input array:", x)
# print("Gradient:", grad)


# import jax
# import jax.numpy as jnp
# from typing import NamedTuple, Callable
# import equinox as eqx


# class Pytree: ...


# class Temp(NamedTuple):
#     x: str
#     z: Callable[[float], int]

#     def test(self):
#         return 4


# # Define the namedtuple subclass
# class MyClass[T](eqx.Module, Pytree):
#     x: T
#     y: str
#     z: Temp


# class Temp2(NamedTuple):

#     def test(self):
#         print("recompiling 2")
#         return 4


# # Function to calculate scalar output
# def func(my_class: MyClass[jax.Array], test: Temp2) -> jax.Array:
#     print("recompiling")
#     return jnp.sum(my_class.x**2 * test.test())


# # Grad function
# # grad_func = eqx.filter_jit(jax.grad(func))


# # Example usage
# # with jax.log_compiles(True):
# x = jnp.ones((5, 5))
# my_instance = MyClass[jnp.ndarray](x, "hi", Temp(1.0, int))
# f_safe = eqx.filter_jit(eqx.filter_grad(func))
# # t = Temp2()
# grad_output = f_safe(my_instance, Temp2())
# grad_output = f_safe(my_instance, Temp2())
# grad_output = f_safe(my_instance, Temp2())

# print(grad_output.x)

# import jax.numpy as jnp
# from typing import NamedTuple
# import jax
# import equinox as eqx


# class Test(NamedTuple):
#     x: jax.Array
#     y: jax.Array


# x = Test(jnp.ones((1, 5)), jnp.ones((1, 5)))
# # x, _ = jax.tree.flatten(x)
# # print(x)
# # quit()
# # print(jax.numpy.linalg.norm(x))


# @eqx.filter_jit
# def pytree_norm(tree):
#     squared = jax.tree.map(lambda x: jnp.sum(x**2), tree)
#     return jnp.sqrt(jax.tree.reduce(jnp.add, squared))


# print(pytree_norm(x))


import jax
import jax.numpy as jnp
from jax import lax
from typing import NamedTuple, Callable
import equinox as eqx


# Define a simple scan function
# This will increment the carry by the current element of xs
# and append the carry to the output sequence.
def scan_fn(carry, tree):
    carry += tree.y.x
    return carry, carry


class Temp(eqx.Module):
    x: int


class Temp2(eqx.Module):
    y: Temp


# Test inputs
test_array = Temp2(Temp(jnp.array([1, 2, 3, 4, 5])))

# Use lax.scan
try:
    final_carry, outputs = lax.scan(scan_fn, 0, test_array)
    print("Final carry:", final_carry)
    print("Outputs:", outputs)
except Exception as e:
    print("Error:", str(e))


# def test[T: int | str](fn: Callable[[T], T]) -> Callable[[T], T]:
#     return fn


# def mip(x: int) -> int:
#     return x


# x = test(mip)
