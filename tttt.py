# from typing import Protocol, TypeVar

# # Define the Protocols
# class RNN_LEARNABLE(Protocol):
#     def train(self) -> None:
#         """Train the model."""
#         pass

# class WithBaseFuture(Protocol):
#     def predict_future(self) -> str:
#         """Predict future outcomes."""
#         pass

# # Combine into a single Protocol-like class
# class _BaseFutureCap(RNN_LEARNABLE, WithBaseFuture, Protocol):
#     pass

# # Define a TypeVar bound to the combined Protocol
# T = TypeVar("T", bound=_BaseFutureCap)

# # Function that uses the TypeVar
# def use_future_cap(obj: T) -> None:
#     obj.train()
#     # print(obj.predict_future())

# # Dummy class that satisfies the Protocol
# class MyModel:
#     def train(self) -> None:
#         print("Training the model...")

#     def predict_future(self) -> str:
#         return "The future looks bright!"

# # Dummy class that does NOT satisfy the Protocol
# class IncompleteModel:
#     def train(self) -> None:
#         print("Training the incomplete model...")

# # Main function to test
# if __name__ == "__main__":
#     model = MyModel()
#     use_future_cap(model)  # Works fine

#     incomplete_model = IncompleteModel()
#     # Uncommenting the line below will cause a mypy error
#     # use_future_cap(incomplete_model)

# import torch
# from torch import vmap


# # Define a function that returns a tuple
# def example_function(x):
#     return x + 1, x * 2


# # Inputs
# inputs = torch.tensor([1.0, 2.0, 3.0])  # Shape: (3,)

# # Apply vmap to the function
# batched_function = vmap(example_function)
# output = batched_function(inputs)

# print(output)


# import torch
# from torch.func import jacfwd

# # Define the input tensor
# x = torch.randn(5)


# # Define function f
# def f(x):
#     return x.sin()


# # Define function g
# def g(x):
#     result = f(x)
#     aux_data = {"input": x, "output": result}  # Return a dictionary as auxiliary data
#     return result, aux_data


# # Compute the Jacobian and auxiliary result
# jacobian_f, aux_data = jacfwd(g, has_aux=True)(x)

# print(type(jacobian_f))
# # Verify that the auxiliary data contains the correct input and output
# print(aux_data)
# assert torch.allclose(aux_data["output"], f(x))

# import torch
# from torch.func import jacrev


# # Define a simple function
# def simple_function(x):
#     return (x**2 + 3 * x + 5).sum()


# # Input tensor with requires_grad=False
# x = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)

# # Compute the Jacobian
# jacobian = jacrev(simple_function)(x)

# print("Input:", x)
# print("Jacobian:", jacobian)


# import torch
# from torch.utils import _pytree as pytree
# from dataclasses import dataclass


# @dataclass(frozen=True)
# class CustomData:
#     value: torch.Tensor  # Batched tensor
#     aux: int  # Static metadata (not batched)


# # Register the PyTree
# def customdata_flatten(custom_data: CustomData):
#     # Flatten into dynamic (value) and static (aux) parts
#     return (custom_data.value,), custom_data.aux


# def customdata_unflatten(children, aux):
#     # Reconstruct from static and dynamic parts
#     return CustomData(value=children[0], aux=aux)


# pytree.register_pytree_node(CustomData, customdata_flatten, customdata_unflatten)


# # Define a function to process CustomData
# def process_custom_data(data: CustomData):
#     # Process only the batched tensor, keep aux unchanged
#     return CustomData(data.value**data.aux, data.aux + 5)


# # Batched input
# values = torch.tensor([[1.0], [2.0], [3.0]])  # Batched data
# aux_value = 2  # Static metadata (not batched)

# # Create a CustomData instance
# batched_data = CustomData(value=values, aux=aux_value)

# # Apply vmap to the function
# result = torch.vmap(process_custom_data)(batched_data)

# print("Result:")
# print(result)


import time
from typing import Callable, Iterable, NamedTuple
import torch
from torch.utils import _pytree as pytree
from dataclasses import dataclass

torch.manual_seed(0)


# def tree_stack(trees: Iterable[pytree.PyTree]) -> pytree.PyTree:
#     return pytree.tree_map(lambda *v: torch.stack(v), *trees)


# def tree_unstack(tree: pytree.PyTree) -> Iterable[pytree.PyTree]:
#     leaves, treedef = pytree.tree_flatten(tree)
#     return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]


# @dataclass(frozen=True)
# class Pytree:
#     a: torch.Tensor

#     def __iter__(self):
#         return iter(tree_unstack(self))


# def customdata_flatten(custom_data: Pytree):
#     return (custom_data.a,), None


# def customdata_unflatten(children, _):
#     return Pytree(a=children[0])


# pytree.register_pytree_node(Pytree, customdata_flatten, customdata_unflatten)

# # Create instances of the dataclass pytree
# pytree1 = Pytree(a=torch.tensor([1, 2]))
# pytree2 = Pytree(a=torch.tensor([7, 8]))

# # Stack them into one list of pytrees
# stacked_pytrees = tree_stack([pytree1, pytree2])

# # Iterate over all leaves
# for leaf in stacked_pytrees:
#     print(leaf)


# @dataclass(frozen=True, slots=True)
# @dataclass


@dataclass(frozen=True)
class Custom:
    b: torch.Tensor
    c: torch.Tensor


def custom_flatten(custom_data: Custom):
    return (custom_data.b, custom_data.c), None


def custom_unflatten(children, aux):
    return Custom(children[0], children[1])


pytree.register_pytree_node(Custom, custom_flatten, custom_unflatten)

# custom_flatten = lambda x: ((x.b,), x.c)
# custum_unflatten = lambda x, y: Custom(x[0], y)
# pytree.register_pytree_node(Custom, custom_flatten, custum_unflatten)


class CustomData(NamedTuple):
    value: torch.Tensor  # Batched tensor
    aux: torch.Tensor  # Static metadata (not batched)
    a: float
    tt: Custom
    # test: Callable[[int], int]

    # def __iter__(self):
    #     return iter(tree_unstack(self))


# Register the PyTree
# def customdata_flatten(custom_data: CustomData):
#     return (custom_data.value, custom_data.aux, custom_data.a), None


# def customdata_unflatten(children, aux):
#     return CustomData(
#         value=children[0],
#         aux=children[1],
#         a=children[2],
#     )


# pytree.register_pytree_node(CustomData, customdata_flatten, customdata_unflatten)


# def tester(data: CustomData):
#     return CustomData(data.value, data.aux + 10)


def process_custom_data(data: CustomData, x: int):
    # Simulate a more expensive computation
    x = data.tt.b * data.aux
    env = CustomData(
        data.value * data.aux + x, data.aux * data.a, data.a + 1, data.tt
    )  # also works!: data.aux + torch.ceil(data.value.mean()).int()
    return env


def fn(mydata: Custom, x: int):
    # Simulate a more expensive computation
    data = CustomData(torch.tensor([1, 2]), torch.tensor([3, 4]), 2.0, mydata)
    x = data.tt.b * data.aux
    env = CustomData(
        data.value * data.aux + x, data.aux * data.a, data.a + 1, Custom(x, data.tt.c)
    )  # also works!: data.aux + torch.ceil(data.value.mean()).int()
    return env.value


# def fn1(data: CustomData, x):
#     shape = CustomData(value=0, aux=0, a=None, tt=Custom(0, None))
#     l = torch.vmap(fn, in_dims=(shape, None))(data, x)
#     return torch.sum(l)


# Batched input
values1 = torch.randn(3, 1)  # Larger batched data for meaningful benchmarking
values2 = torch.randn(3, 1)
aux_value1 = torch.randn(3, 1)
aux_value2 = torch.randn(3, 1)
temp = Custom(torch.randn(3, 1), torch.randn(3, 1))


batched_data = CustomData(value=values1, aux=aux_value1, a=2.0, tt=temp)

ttttt = CustomData(value=0, aux=0, a=None, tt=0)

# Measure vmap execution time
start_time_vmap = time.time()
result_vmap = torch.vmap(process_custom_data, in_dims=(ttttt, None), out_dims=ttttt)(
    batched_data, 1
)
vmap_time = time.time() - start_time_vmap


print(f"vmap execution time: {vmap_time:.6f} seconds")

print(result_vmap)
print(result_vmap.value.shape)
print(result_vmap.aux.shape)

# jac = torch.func.jacrev(fn)(temp, 1)
# print(jac)

# # Measure manual loop execution time
# start_time_loop = time.time()
# result_loop = manual_loop(batched_data)
# loop_time = time.time() - start_time_loop

# # Validate correctness
# assert torch.allclose(result_vmap.value, result_loop.value)
# assert result_vmap.aux == result_loop.aux
# print(f"Manual loop execution time: {loop_time:.6f} seconds")

# print(result_vmap)
# for leaf in result_vmap:
#     print(leaf)
# print(num)


# for leaf in result_vmap:
#     print(leaf)

# for leaf in snd_result:
#     print(leaf)


# from typing import Protocol, Self, TypeVar
# from pyrsistent import PClass, field


# class TempA(PClass):
#     a: int = field(type=int, mandatory=True)
#     b: int = field(type=int, mandatory=True)


# class SetProtocol(Protocol):
#     def set(self, *args, **kwargs) -> Self:
#         pass


# T = TypeVar("T", bound=SetProtocol)


# def test(d: T):
#     c = d.set(a=1)


# x = TempA(a=1, b=2)
# test(x)

# from operator import add


# def foldr(f):
#     def foldr_(xs, x):
#         result = x
#         for item in xs:
#             result = f(item, result)
#         return result

#     return foldr_


# import cProfile, pstats


# profiler = cProfile.Profile()
# profiler.enable()
# foldr(add)(range(100000), 0)
# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats("cumtime")
# stats.print_stats()


# from dataclasses import dataclass
# from typing import Generic, TypeVar
# import torch
# from torch.utils import _pytree as pytree

# T = TypeVar("T")


# @dataclass(frozen=True)
# class MyPyTree(Generic[T]):
#     x: T
#     y: list[torch.Tensor]


# # Register the class as a PyTree
# def tree_flatten(obj: MyPyTree[T]):
#     # Extract the tensors as a list and return auxiliary data
#     return (obj.x, obj.y), None


# def tree_unflatten(children, aux):
#     # Reconstruct the object from its flattened representation
#     return MyPyTree[T](*children)


# pytree.register_pytree_node(MyPyTree, tree_flatten, tree_unflatten)


# # Define a function to test differentiability
# def func(pytree: list[MyPyTree[torch.Tensor]]):
#     return (pytree[0].x ** 2).sum()


# # Create an instance of MyPyTree with tensors
# x = torch.tensor([1.0, 2.0])
# p = MyPyTree[torch.Tensor](x, [torch.tensor([3.0, 4.0])])

# # Compute the Jacobian using jacrev
# print(p)
# jacobian = torch.func.jacrev(func)([p])

# # Print the Jacobian to verify
# print(jacobian)

# # j, tree = pytree.tree_flatten(jacobian)
# tree = pytree.tree_structure([p])
# zeroed = pytree.tree_unflatten((torch.tensor(0),) * tree.num_leaves, tree)


# print(pytree.tree_map(lambda x, y: x + y, jacobian, zeroed))
# print(jacobian + torch.tensor(0))


# from typing import Generic, TypeVar

# T = TypeVar("T")


# class Test(Generic[T]):
#     pass


# TEST = TypeVar("TEST", bound=Test, contravariant=True)


# def test(x: TEST) -> None:
#     pass


# t = Test[int]()
# test(t)


# from dataclasses import dataclass
# from typing import Generic, Protocol, TypeVar

# A = TypeVar("A")
# B = TypeVar("B")


# class WithA(Protocol[A]):
#     a: A


# class WithB(Protocol[B]):
#     b: B


# class WithAB(WithA[A], WithB[B], Protocol[A, B]):
#     pass


# @dataclass(frozen=True)
# class Test(WithAB[A, B]):
#     a: A
#     b: B


# CHECK = TypeVar("CHECK", bound=WithAB[int, int])


# class TestMe(Generic[CHECK]):
#     def test(self, x: CHECK) -> None:
#         pass


# x = Test[int, int](1, 2)

# TestMe[Test[int, int]]().test(x)


# from typing import Any, Generic, NamedTuple, Protocol, Self, TypeVar
# import typing_extensions


# class WithA(Protocol):
#     @property
#     def a(self) -> int: ...

#     # def __replace__(self, /, **changes):
#     #     pass
#     def __replace__(self, **kwargs: Any) -> typing_extensions.Self: ...


# A = TypeVar("A")
# B = TypeVar("B")


# class Test(NamedTuple, Generic[A, B]):
#     a: A
#     b: B


# T = TypeVar("T", bound=WithA)


# def test(x: WithA):
#     pass


# x = Test[int, int](1, 2)
# x._replace

# y = test(x)


# import cProfile
# import pstats
# from collections import namedtuple
# import copy

# # Define a named tuple
# TreeStack = namedtuple("TreeStack", ["field1", "field2", "field3"])


# # Function to simulate training steps
# def trainStep(data):
#     for _ in range(100_000):
#         # Call copy._replace (equivalent to .replace in namedtuple)
#         # data = data._replace(field1=data.field1 + 1)
#         data = copy.replace(data, field1=data.field1 + 1)
#         data = copy.replace(data, field1=data.field1 + 1)
#         data = copy.replace(data, field1=data.field1 + 1)
#         data = copy.replace(data, field1=data.field1 + 1)
#         data = copy.replace(data, field1=data.field1 + 1)
#         data = copy.replace(data, field1=data.field1 + 1)
#         data = copy.replace(data, field1=data.field1 + 1)
#         data = copy.replace(data, field1=data.field1 + 1)
#         data = copy.replace(data, field1=data.field1 + 1)
#         data = copy.replace(data, field1=data.field1 + 1)
#         data = copy.replace(data, field1=data.field1 + 1)
#     return data


# # Prepare data
# data = TreeStack(field1=0, field2=0, field3=0)

# # Profile the function
# profiler = cProfile.Profile()
# profiler.enable()
# predictions = trainStep(data)
# profiler.disable()

# # Output and save profiling stats
# stats = pstats.Stats(profiler).sort_stats("cumtime")
# stats.print_stats()
# stats.dump_stats("profile_results.prof")
