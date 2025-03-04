# # from typing import Protocol, TypeVar

# # # Define the Protocols
# # class RNN_LEARNABLE(Protocol):
# #     def train(self) -> None:
# #         """Train the model."""
# #         pass

# # class WithBaseFuture(Protocol):
# #     def predict_future(self) -> str:
# #         """Predict future outcomes."""
# #         pass

# # # Combine into a single Protocol-like class
# # class _BaseFutureCap(RNN_LEARNABLE, WithBaseFuture, Protocol):
# #     pass

# # # Define a TypeVar bound to the combined Protocol
# # T = TypeVar("T", bound=_BaseFutureCap)

# # # Function that uses the TypeVar
# # def use_future_cap(obj: T) -> None:
# #     obj.train()
# #     # print(obj.predict_future())

# # # Dummy class that satisfies the Protocol
# # class MyModel:
# #     def train(self) -> None:
# #         print("Training the model...")

# #     def predict_future(self) -> str:
# #         return "The future looks bright!"

# # # Dummy class that does NOT satisfy the Protocol
# # class IncompleteModel:
# #     def train(self) -> None:
# #         print("Training the incomplete model...")

# # # Main function to test
# # if __name__ == "__main__":
# #     model = MyModel()
# #     use_future_cap(model)  # Works fine

# #     incomplete_model = IncompleteModel()
# #     # Uncommenting the line below will cause a mypy error
# #     # use_future_cap(incomplete_model)

# # import torch
# # from torch import vmap


# # # Define a function that returns a tuple
# # def example_function(x):
# #     return x + 1, x * 2


# # # Inputs
# # inputs = torch.tensor([1.0, 2.0, 3.0])  # Shape: (3,)

# # # Apply vmap to the function
# # batched_function = vmap(example_function)
# # output = batched_function(inputs)

# # print(output)


# # import torch
# # from torch.func import jacfwd

# # # Define the input tensor
# # x = torch.randn(5)


# # # Define function f
# # def f(x):
# #     return x.sin()


# # # Define function g
# # def g(x):
# #     result = f(x)
# #     aux_data = {"input": x, "output": result}  # Return a dictionary as auxiliary data
# #     return result, aux_data


# # # Compute the Jacobian and auxiliary result
# # jacobian_f, aux_data = jacfwd(g, has_aux=True)(x)

# # print(type(jacobian_f))
# # # Verify that the auxiliary data contains the correct input and output
# # print(aux_data)
# # assert torch.allclose(aux_data["output"], f(x))

# # import torch
# # from torch.func import jacrev


# # # Define a simple function
# # def simple_function(x):
# #     return (x**2 + 3 * x + 5).sum()


# # # Input tensor with requires_grad=False
# # x = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)

# # # Compute the Jacobian
# # jacobian = jacrev(simple_function)(x)

# # print("Input:", x)
# # print("Jacobian:", jacobian)


# # import torch
# # from torch.utils import _pytree as pytree
# # from dataclasses import dataclass


# # @dataclass(frozen=True)
# # class CustomData:
# #     value: torch.Tensor  # Batched tensor
# #     aux: int  # Static metadata (not batched)


# # # Register the PyTree
# # def customdata_flatten(custom_data: CustomData):
# #     # Flatten into dynamic (value) and static (aux) parts
# #     return (custom_data.value,), custom_data.aux


# # def customdata_unflatten(children, aux):
# #     # Reconstruct from static and dynamic parts
# #     return CustomData(value=children[0], aux=aux)


# # pytree.register_pytree_node(CustomData, customdata_flatten, customdata_unflatten)


# # # Define a function to process CustomData
# # def process_custom_data(data: CustomData):
# #     # Process only the batched tensor, keep aux unchanged
# #     return CustomData(data.value**data.aux, data.aux + 5)


# # # Batched input
# # values = torch.tensor([[1.0], [2.0], [3.0]])  # Batched data
# # aux_value = 2  # Static metadata (not batched)

# # # Create a CustomData instance
# # batched_data = CustomData(value=values, aux=aux_value)

# # # Apply vmap to the function
# # result = torch.vmap(process_custom_data)(batched_data)

# # print("Result:")
# # print(result)


# import time
# from typing import Callable, Iterable, NamedTuple
# import torch
# from torch.utils import _pytree as pytree
# from dataclasses import dataclass

# torch.manual_seed(0)


# # def tree_stack(trees: Iterable[pytree.PyTree]) -> pytree.PyTree:
# #     return pytree.tree_map(lambda *v: torch.stack(v), *trees)


# # def tree_unstack(tree: pytree.PyTree) -> Iterable[pytree.PyTree]:
# #     leaves, treedef = pytree.tree_flatten(tree)
# #     return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]


# # @dataclass(frozen=True)
# # class Pytree:
# #     a: torch.Tensor

# #     def __iter__(self):
# #         return iter(tree_unstack(self))


# # def customdata_flatten(custom_data: Pytree):
# #     return (custom_data.a,), None


# # def customdata_unflatten(children, _):
# #     return Pytree(a=children[0])


# # pytree.register_pytree_node(Pytree, customdata_flatten, customdata_unflatten)

# # # Create instances of the dataclass pytree
# # pytree1 = Pytree(a=torch.tensor([1, 2]))
# # pytree2 = Pytree(a=torch.tensor([7, 8]))

# # # Stack them into one list of pytrees
# # stacked_pytrees = tree_stack([pytree1, pytree2])

# # # Iterate over all leaves
# # for leaf in stacked_pytrees:
# #     print(leaf)


# # @dataclass(frozen=True, slots=True)
# # @dataclass


# @dataclass(frozen=True)
# class Custom:
#     b: torch.Tensor
#     c: torch.Tensor


# def custom_flatten(custom_data: Custom):
#     return (custom_data.b, custom_data.c), None


# def custom_unflatten(children, aux):
#     return Custom(children[0], children[1])


# pytree.register_pytree_node(Custom, custom_flatten, custom_unflatten)

# # custom_flatten = lambda x: ((x.b,), x.c)
# # custum_unflatten = lambda x, y: Custom(x[0], y)
# # pytree.register_pytree_node(Custom, custom_flatten, custum_unflatten)


# class CustomData(NamedTuple):
#     value: torch.Tensor  # Batched tensor
#     aux: torch.Tensor  # Static metadata (not batched)
#     a: float
#     tt: Custom
#     # test: Callable[[int], int]

#     # def __iter__(self):
#     #     return iter(tree_unstack(self))


# # Register the PyTree
# # def customdata_flatten(custom_data: CustomData):
# #     return (custom_data.value, custom_data.aux, custom_data.a), None


# # def customdata_unflatten(children, aux):
# #     return CustomData(
# #         value=children[0],
# #         aux=children[1],
# #         a=children[2],
# #     )


# # pytree.register_pytree_node(CustomData, customdata_flatten, customdata_unflatten)


# # def tester(data: CustomData):
# #     return CustomData(data.value, data.aux + 10)


# def process_custom_data(data: CustomData, x: int):
#     # Simulate a more expensive computation
#     x = data.tt.b * data.aux
#     env = CustomData(
#         data.value * data.aux + x, data.aux * data.a, data.a + 1, data.tt
#     )  # also works!: data.aux + torch.ceil(data.value.mean()).int()
#     return env


# def fn(mydata: Custom, x: int):
#     # Simulate a more expensive computation
#     data = CustomData(torch.tensor([1, 2]), torch.tensor([3, 4]), 2.0, mydata)
#     x = data.tt.b * data.aux
#     env = CustomData(
#         data.value * data.aux + x, data.aux * data.a, data.a + 1, Custom(x, data.tt.c)
#     )  # also works!: data.aux + torch.ceil(data.value.mean()).int()
#     return env.value


# # def fn1(data: CustomData, x):
# #     shape = CustomData(value=0, aux=0, a=None, tt=Custom(0, None))
# #     l = torch.vmap(fn, in_dims=(shape, None))(data, x)
# #     return torch.sum(l)


# # Batched input
# values1 = torch.randn(3, 1)  # Larger batched data for meaningful benchmarking
# values2 = torch.randn(3, 1)
# aux_value1 = torch.randn(3, 1)
# aux_value2 = torch.randn(3, 1)
# temp = Custom(torch.randn(3, 1), torch.randn(3, 1))


# batched_data = CustomData(value=values1, aux=aux_value1, a=2.0, tt=temp)

# ttttt = CustomData(value=0, aux=0, a=None, tt=0)

# # Measure vmap execution time
# start_time_vmap = time.time()
# result_vmap = torch.vmap(process_custom_data, in_dims=(ttttt, None), out_dims=ttttt)(
#     batched_data, 1
# )
# vmap_time = time.time() - start_time_vmap


# print(f"vmap execution time: {vmap_time:.6f} seconds")

# print(result_vmap)
# print(result_vmap.value.shape)
# print(result_vmap.aux.shape)

# # jac = torch.func.jacrev(fn)(temp, 1)
# # print(jac)

# # # Measure manual loop execution time
# # start_time_loop = time.time()
# # result_loop = manual_loop(batched_data)
# # loop_time = time.time() - start_time_loop

# # # Validate correctness
# # assert torch.allclose(result_vmap.value, result_loop.value)
# # assert result_vmap.aux == result_loop.aux
# # print(f"Manual loop execution time: {loop_time:.6f} seconds")

# # print(result_vmap)
# # for leaf in result_vmap:
# #     print(leaf)
# # print(num)


# # for leaf in result_vmap:
# #     print(leaf)

# # for leaf in snd_result:
# #     print(leaf)


# # from typing import Protocol, Self, TypeVar
# # from pyrsistent import PClass, field


# # class TempA(PClass):
# #     a: int = field(type=int, mandatory=True)
# #     b: int = field(type=int, mandatory=True)


# # class SetProtocol(Protocol):
# #     def set(self, *args, **kwargs) -> Self:
# #         pass


# # T = TypeVar("T", bound=SetProtocol)


# # def test(d: T):
# #     c = d.set(a=1)


# # x = TempA(a=1, b=2)
# # test(x)

# # from operator import add


# # def foldr(f):
# #     def foldr_(xs, x):
# #         result = x
# #         for item in xs:
# #             result = f(item, result)
# #         return result

# #     return foldr_


# # import cProfile, pstats


# # profiler = cProfile.Profile()
# # profiler.enable()
# # foldr(add)(range(100000), 0)
# # profiler.disable()
# # stats = pstats.Stats(profiler).sort_stats("cumtime")
# # stats.print_stats()


# # from dataclasses import dataclass
# # from typing import Generic, TypeVar
# # import torch
# # from torch.utils import _pytree as pytree

# # T = TypeVar("T")


# # @dataclass(frozen=True)
# # class MyPyTree(Generic[T]):
# #     x: T
# #     y: list[torch.Tensor]


# # # Register the class as a PyTree
# # def tree_flatten(obj: MyPyTree[T]):
# #     # Extract the tensors as a list and return auxiliary data
# #     return (obj.x, obj.y), None


# # def tree_unflatten(children, aux):
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
# # print(jacobian + torch.tensor(0))


# # from typing import Generic, TypeVar

# # T = TypeVar("T")


# # class Test(Generic[T]):
# #     pass


# # TEST = TypeVar("TEST", bound=Test, contravariant=True)


# # def test(x: TEST) -> None:
# #     pass


# # t = Test[int]()
# # test(t)


# # from dataclasses import dataclass
# # from typing import Generic, Protocol, TypeVar

# # A = TypeVar("A")
# # B = TypeVar("B")


# # class WithA(Protocol[A]):
# #     a: A


# # class WithB(Protocol[B]):
# #     b: B


# # class WithAB(WithA[A], WithB[B], Protocol[A, B]):
# #     pass


# # @dataclass(frozen=True)
# # class Test(WithAB[A, B]):
# #     a: A
# #     b: B


# # CHECK = TypeVar("CHECK", bound=WithAB[int, int])


# # class TestMe(Generic[CHECK]):
# #     def test(self, x: CHECK) -> None:
# #         pass


# # x = Test[int, int](1, 2)

# # TestMe[Test[int, int]]().test(x)


# # from typing import Any, Generic, NamedTuple, Protocol, Self, TypeVar
# # import typing_extensions


# # class WithA(Protocol):
# #     @property
# #     def a(self) -> int: ...

# #     # def __replace__(self, /, **changes):
# #     #     pass
# #     def __replace__(self, **kwargs: Any) -> typing_extensions.Self: ...


# # A = TypeVar("A")
# # B = TypeVar("B")


# # class Test(NamedTuple, Generic[A, B]):
# #     a: A
# #     b: B


# # T = TypeVar("T", bound=WithA)


# # def test(x: WithA):
# #     pass


# # x = Test[int, int](1, 2)
# # x._replace

# # y = test(x)


# # import cProfile
# # import pstats
# # from collections import namedtuple
# # import copy

# # # Define a named tuple
# # TreeStack = namedtuple("TreeStack", ["field1", "field2", "field3"])


# # # Function to simulate training steps
# # def trainStep(data):
# #     for _ in range(100_000):
# #         # Call copy._replace (equivalent to .replace in namedtuple)
# #         # data = data._replace(field1=data.field1 + 1)
# #         data = copy.replace(data, field1=data.field1 + 1)
# #         data = copy.replace(data, field1=data.field1 + 1)
# #         data = copy.replace(data, field1=data.field1 + 1)
# #         data = copy.replace(data, field1=data.field1 + 1)
# #         data = copy.replace(data, field1=data.field1 + 1)
# #         data = copy.replace(data, field1=data.field1 + 1)
# #         data = copy.replace(data, field1=data.field1 + 1)
# #         data = copy.replace(data, field1=data.field1 + 1)
# #         data = copy.replace(data, field1=data.field1 + 1)
# #         data = copy.replace(data, field1=data.field1 + 1)
# #         data = copy.replace(data, field1=data.field1 + 1)
# #     return data


# # # Prepare data
# # data = TreeStack(field1=0, field2=0, field3=0)

# # # Profile the function
# # profiler = cProfile.Profile()
# # profiler.enable()
# # predictions = trainStep(data)
# # profiler.disable()

# # # Output and save profiling stats
# # stats = pstats.Stats(profiler).sort_stats("cumtime")
# # stats.print_stats()
# # stats.dump_stats("profile_results.prof")

# x = "hi"


# import jax
# import jax.numpy as jnp
# import equinox as eqx
# from typing import Optional
# from pyrsistent import pvector
# from pyrsistent.typing import PVector


# class Log(eqx.Module):
#     loss: Optional[jax.Array] = eqx.field(default=None)
#     something_else: Optional[jax.Array] = eqx.field(default=None)


# class MyModule(eqx.Module):
#     weight: jnp.ndarray
#     log: Log


# def loss_and_update_fn(module: MyModule, x: jnp.ndarray) -> tuple[float, MyModule]:
#     print("recompiled")
#     # Use the non-auxiliary `weight` to compute the loss
#     loss = jnp.sum((x @ module.weight) ** 2)

#     # Append something to the auxiliary list
#     log = Log(loss=loss)

#     return loss, eqx.tree_at(lambda t: t.log, module, eqx.combine(log, module.log))


# # Example setup
# module = MyModule(
#     weight=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
#     log=Log(
#         something_else=jnp.array([1.0, 2.0]), loss=jnp.array(0.0, dtype=jnp.float32)
#     ),
# )
# x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

# # Use filter_jacfwd with has_aux=True to compute the jacobian and pass the auxiliary data
# jacobian_fn = eqx.filter_jit(
#     lambda m, _a: eqx.filter_jacfwd(lambda a: loss_and_update_fn(m, a), has_aux=True)(
#         _a
#     )
# )

# # Compute the Jacobian and updated module
# jacobian, updated_module = jacobian_fn(module, x)
# jacobian, updated_module = jacobian_fn(updated_module, x)

# # Outputs
# print("Jacobian:", jacobian)
# print("Updated module aux_list:", updated_module.log.loss)

# # from typing import Optional
# # import equinox as eqx
# # import jax


# # class Test(eqx.Module):
# #     a: Optional[jax.Array] = eqx.field(default=None)
# #     b: Optional[jax.Array] = eqx.field(default=None)


# # model = Test(a=jax.numpy.array([1, 2]), b=jax.numpy.array([3, 4]))
# # update = Test(a=jax.numpy.array([4, 4]))
# # updated = eqx.combine(update, model)
# # print(updated.a)
# # print(updated.b)


# import jax
# import jax.numpy as jnp
# import equinox as eqx


# # (1) Define a constant function that returns a constant pytree
# def constant_pytree_fn():
#     return {"weights": jnp.array([1.0, 2.0, 3.0]), "bias": jnp.array(0.5)}


# # (2) Define the scan function
# def scan_fn(carry, _):
#     print("Recompil;ed")
#     # Unpack the carry
#     value1, value2 = carry

#     value1 = eqx.apply_updates(constant_pytree_fn(), value1)

#     # Update value2 (for demonstration, we'll increment it)
#     value2 += 1

#     # Return the updated carry and no predictions
#     return (value1, value2), None


# # Initial carry values: value1 is None, value2 is 0
# initial_carry = (jax.tree.map(lambda x: jnp.array(0), constant_pytree_fn()), 0)

# # Number of steps to scan
# num_steps = 5

# # Perform the scan
# final_carry, _ = jax.lax.scan(scan_fn, initial_carry, None, length=num_steps)

# # Unpack the final carry
# final_value1, final_value2 = final_carry

# # Print the results
# print("Final value1:", final_value1)
# print("Final value2:", final_value2)


# import jax
# import jax.numpy as jnp


# # Define a function to scan over
# def accumulate_sum(carry, x):
#     print("recompiled")
#     carry = carry + x
#     return carry, carry  # carry is the new state, second output accumulates results


# # JIT-compile the scan
# # @jax.jit
# def run_scan(array):
#     # print("recompiled1")

#     initial_carry = 0.0
#     return jax.lax.scan(accumulate_sum, initial_carry, array)


# # Initialize an array
# array = jnp.array([1, 2, 3, 4, 5], dtype=jnp.float32)

# # Execute the JIT-compiled scan
# final_carry, accumulated_sums = run_scan(array)
# final_carry, accumulated_sums = run_scan(array)

# # Display the results
# print("Final carry (sum of array):", final_carry)
# print("Accumulated sums:", accumulated_sums)


# import jax.numpy as jnp
# import jax
# import equinox as eqx

# # Input array and truncation size
# x = jnp.arange(18)
# t = 5  # Truncation size

# # Compute how many elements form complete chunks of size T


# def test(T: int):
#     print("recompiled")
#     n_complete = (len(x) // T) * T

#     # Split the array
#     main_part = x[:n_complete]  # Main part that can be reshaped
#     extra_part = x[n_complete:]  # Leftover elements

#     # Reshape the main part
#     reshaped = main_part.reshape(-1, T)
#     return reshaped, extra_part


# fn = eqx.filter_jit(test)
# reshaped, extra_part = fn(t)
# reshaped, extra_part = fn(t)
# reshaped, extra_part = fn(t)
# reshaped, extra_part = fn(t)

# # Print results
# print("Reshaped array:")
# print(reshaped)
# print("Leftover array:")
# print(extra_part)


# from dataclasses import dataclass
# import jax
# import jax.numpy as jnp
# import equinox as eqx

# # Example PyTree structure
# pytree = {
#     "a": jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
#     "b": jnp.array([10, 11, 12, 13, 14, 15, 16, 17]),
# }
# T = 3  # Truncation size


# @dataclass
# class SplitResult:
#     reshaped: jax.Array
#     leftover: jax.Array


# def split_and_reshape(arr: jax.Array, T: int):
#     n_complete = (len(arr) // T) * T
#     main_part = arr[:n_complete].reshape((-1, T) + arr.shape[1:])
#     extra_part = arr[n_complete:]  # Leftover
#     return SplitResult(main_part, extra_part)


# def pytree_split[Tree: eqx.Module](tree: Tree, T: int) -> tuple[Tree, Tree]:
#     reshaped_pytree_: Tree = jax.tree.map(lambda arr: split_and_reshape(arr, T), tree)
#     reshaped_pytree = jax.tree.map(lambda x: x.reshaped, reshaped_pytree_)
#     leftover_pytree = jax.tree.map(lambda x: x.leftover, reshaped_pytree_)
#     return reshaped_pytree, leftover_pytree


# pytree2 = {
#     "a": jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
#     "b": jnp.array(
#         [
#             [100, 101],
#             [100, 101],
#             [100, 101],
#             [100, 101],
#             [100, 101],
#             [100, 101],
#             [100, 101],
#             [100, 101],
#             [100, 101],
#             [100, 101],
#         ]
#     ),
# }

# reshaped_pytree, leftover_pytree = eqx.filter_jit(pytree_split)(pytree, T)
# print(reshaped_pytree)
# reshaped_pytree, leftover_pytree = eqx.filter_jit(pytree_split)(pytree, T)
# print(reshaped_pytree)
# reshaped_pytree, leftover_pytree = eqx.filter_jit(pytree_split)(pytree2, T)

# # Print the results
# print("Reshaped PyTree:")
# print(reshaped_pytree["b"].shape)
# print("\nLeftover PyTree:")
# print(leftover_pytree)


# import jax
# import jax.numpy as jnp
# from jax import tree_util
# import equinox as eqx


# # Define a function connecting PyTree2 to PyTree1
# def f(input_pytree):
#     x, y = input_pytree["x"], input_pytree["y"]
#     return {"z": x**2 + y, "w": x * y}


# # Input PyTree (PyTree2)
# input_pytree = {"x": jnp.array([2.0, 3.0]), "y": jnp.array([44.0, 2.0])}

# # Compute the output PyTree (PyTree1)
# output_pytree = f(input_pytree)

# # Compute Jacobian of the output PyTree w.r.t. the input PyTree
# jacobian_fn = jax.jacobian(f)

# # Jacobian of the function
# jacobian = jacobian_fn(input_pytree)

# eqx.tree_pprint(jacobian)

# print(jacobian["w"]["x"])

# # # Define a function taking a PyTree as input
# # def h(pytree):
# #     x, y = pytree["x"], pytree["y"]
# #     return {"z": x**2 + y, "w": x * y}


# # # PyTree input and tangent
# # primal = {"x": jnp.array(2.0), "y": jnp.array(3.0)}
# # tangent = {"x": jnp.array(1.0), "y": jnp.array(0.0)}

# # # Compute JVP
# # primal_out, tangent_out = jax.jvp(h, (primal,), (tangent,))
# # print("Primal output:", primal_out)  # h(pytree)
# # print("Tangent output:", tangent_out)  # Directional derivative


# import jax
# import jax.numpy as jnp

# # Example pytrees
# pytree1 = {"a": jnp.array([1, 2, 3]), "b": jnp.array([4, 5, 6])}


# pytree2 = {
#     "a": {"x": jnp.array([40, 50, 60]), "y": jnp.array([70, 80, 90])},
#     "b": {"x": jnp.array([40, 50, 60]), "y": jnp.array([70, 80, 90])},
# }


# # Function to apply element-wise to each leaf
# def add_arrays(arr1, arr2):
#     print(arr1)
#     print(arr2)
#     return arr1 + arr2


# # Using jax.tree_map to apply the function over the pytrees
# result = jax.tree_map(add_arrays, pytree1, jnp.array([1, 2, 3]))

# print(result)


# import tree_math as tm
# import jax.numpy as jnp
# import jax
# import equinox as eqx


# def compute_fn(v):
#     return jax.tree.map(lambda leaf: leaf**2, v)


# # Wrap with JAX's jacobian
# compute_fn_jacobian = jax.jacobian(compute_fn)

# # Example usage
# v = tm.Vector(
#     {"x": jnp.arange(2, dtype=jnp.float32), "y": jnp.arange(5, dtype=jnp.float32)}
# )

# # Compute the function value and its Jacobian
# value = compute_fn(v)
# jacobian = compute_fn_jacobian(v)

# print("Function value:", value)
# eqx.tree_pprint(jacobian.tree)

# tangent = tm.Vector(
#     {"x": jnp.arange(2, dtype=jnp.float32), "y": jnp.arange(5, dtype=jnp.float32)}
# )

# print(tangent @ jacobian)

# # Loop through the first batch dimension
# for i in range(jacobian.tree["x"].tree["x"].shape[0]):  # Assuming batch size is the first dimension
#     single_pytree = jax.tree_util.tree_map(lambda x: x[i], pytree)
#     print(f"PyTree for batch {i}: {single_pytree}")


# import jax
# import jax.numpy as jnp

# # Define pytree1 where each key points to a simple structure (not deeply recursive)
# pytree1 = {
#     "a": {"a": jnp.array([1, 2]), "b": jnp.array([3, 4])},
#     "b": {"a": jnp.array([5, 6]), "b": jnp.array([7, 8])},
# }

# # Define pytree2, which has the same structure but values can be processed differently
# pytree2 = {
#     "a": {
#         "x": jnp.array([10, 20]),
#         "y": jnp.array([30, 40]),
#         "c": jnp.array([50, 600]),
#     },
#     "b": {
#         "x": jnp.array([50, 60]),
#         "y": jnp.array([70, 80]),
#         "c": jnp.array([90, 100]),
#     },
# }


# # Define a function to combine values
# def combine_fn(x, y):
#     print(x)
#     print(y)
#     print()
#     return x


# # Use jax.tree_map to map the function across both pythrees
# result = jax.tree_map(combine_fn, pytree1, pytree2)

# print("Result:")
# print(result)

# from typing import Callable
# import jax
# import jax.numpy as jnp
# import equinox as eqx


# # Define a custom Equinox module to mirror the pytree structure
# class MyModule(eqx.Module):
#     a: "MyModule | jnp.ndarray"  # Nested structure with flexibility for arrays or further modules
#     fn: Callable[[str], str] = eqx.field(default=lambda x: x, static=True)


# # Initialize the Equinox module with the same structure as the pytree
# module = MyModule(a=MyModule(a=jnp.array([1.0, 2.0])))


# # Define a function that takes the Equinox module and returns another module of the same structure
# def my_function(mod: MyModule) -> MyModule:
#     x = mod.a.a
#     # Return a module with the same structure and transformed values
#     return MyModule(a=MyModule(a=x**2))


# # Compute the Jacobian of the function with respect to the module
# jacobian_fn = jax.jacobian(my_function)

# # Apply the Jacobian to the module
# jacobian_result = jacobian_fn(module)

# # Print the Jacobian result
# print("Jacobian result:", jacobian_result)

# # Verify the structure matches the module
# structure_matches = eqx.tree_equal(jacobian_result, module)
# print("Structure matches module:", structure_matches)


# class Head[T]:
#     value: int


# class Node[T]:
#     value: T


# type llist[T] = Head[T] | Node[llist[T]]

# import jax
# import jax.numpy as jnp
# from jax import lax

# # Define a pytree with the same leading dimension but different rest dimensions
# pytree = {
#     "a": jnp.ones((5, 3)),  # Shape (5, 3)
#     "b": None,  # Shape (5, 2) -- Different rest dimension
# }


# # Define scan function
# def scan_fn(carry, x):
#     carry = carry + jnp.sum(x["b"])
#     return carry, carry


# # Run scan
# init = jnp.array(0.0)
# _, out = lax.scan(scan_fn, init, pytree)

# print(out)


# import jax
# import jax.numpy as jnp
# import optax


# # Dummy loss function
# def loss_fn(params, x):
#     return jnp.sum(jnp.square(params * x))


# # Dummy data
# x = jnp.array([1.0, 2.0, 3.0])
# params = jnp.array([0.5, -0.2, 0.8])

# # Compute gradients
# grad_fn = jax.grad(loss_fn)
# grads = grad_fn(params, x)

# # Apply gradient clipping
# clipping_value = 1.0
# updates, _ = optax.clip_by_global_norm(clipping_value).update(grads, optax.EmptyState())

# print("Original Gradients:", grads)
# print("Clipped Gradients:", updates)
# print("gradient norm", 1 / jnp.linalg.norm(grads))
# print("new gradient norm", jnp.linalg.norm(updates))


# from typing import Callable
# import jax
# import jax.numpy as jnp
# import optax
# import equinox as eqx

# jax.config.update("jax_enable_x64", True)


# class IsVector[T](eqx.Module):
#     vector: jax.Array
#     toParam: Callable[[jax.Array], T] = eqx.field(static=True)


# def endowVector[T](tree: T) -> IsVector[T]:
#     vector, toParam = jax.flatten_util.ravel_pytree(tree)
#     return IsVector(vector=vector, toParam=toParam)


# def toVector[T](isVector: IsVector[T]) -> jax.Array:
#     return isVector.vector


# def toParam[T](isVector: IsVector[T]) -> T:
#     return isVector.toParam(isVector.vector)


# # Simple quadratic loss
# def loss_fn(params):
#     return jnp.sum(params**2)


# # Meta-loss function (evaluated after training)
# def meta_loss_fn(params):
#     return jnp.sum((params - 1.0) ** 2)


# # Train function with optimizer unrolling
# def train(params, opt_state, lr, steps=10):
#     opt = optax.adam(lr)
#     for _ in range(steps):
#         grads = jax.grad(loss_fn)(params)
#         updates, opt_state = opt.update(grads, opt_state, params)
#         params = optax.apply_updates(params, updates)
#         jax.debug.print("{}", toVector(endowVector(opt_state)))
#     return params, opt_state


# lr = 0.1
# params = jnp.array([1.0, -1.0])
# opt_state = optax.EmptyState()
# meta_optimize = lambda lr_: meta_loss_fn(train(params, opt_state, lr_, steps=10)[0])

# # Compute autodiff gradient
# lr_grad_autodiff = jax.grad(meta_optimize)(0.1)

# # Compute finite difference approximation
# eps = 1e-4
# meta_loss_plus = meta_optimize(0.1 + eps)
# meta_loss_minus = meta_optimize(0.1 - eps)
# lr_grad_finite_diff = (meta_loss_plus - meta_loss_minus) / (2 * eps)

# # Print results
# print(f"Autodiff gradient: {lr_grad_autodiff}")
# print(f"Finite difference gradient: {lr_grad_finite_diff}")
# print(f"Relative difference: {abs(lr_grad_autodiff - lr_grad_finite_diff) / abs(lr_grad_finite_diff)}")

# opt_state_vec = endowVector(opt_state)
# opt_fn = lambda vec: toVector(endowVector(train(params, opt_state_vec.toParam(vec), lr, steps=10)[1]))
# jacobian = jax.jacobian(opt_fn)(toVector(opt_state_vec))

# jnp.set_printoptions(precision=3)
# jax.debug.print("\n{}", jacobian)
# print(opt_state)


# from jax import numpy as jnp
# import jax
# from typing import NamedTuple, Tuple


# class Nothing(NamedTuple): ...


# class Just[T](NamedTuple):
#     a: T


# type Maybe[T] = Just[T] | Nothing


# def process_maybe(prev_state: float, current_value: float) -> Tuple[float, Maybe[float]]:
#     # if current_value == 0:
#     #     return prev_state, Nothing()
#     # else:
#     #     new_value = prev_state + current_value
#     #     return new_value, Just(new_value)

#     return jax.lax.cond(
#         current_value == 0, lambda x: (x, None), lambda x: (x + current_value, x + current_value), prev_state
#     )


# # The scan function that uses process_maybe
# def scan_example(maybe_values: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
#     init_state = 0.0
#     final_state, outputs = jax.lax.scan(process_maybe, init_state, maybe_values)
#     return final_state, outputs


# # Example usage
# maybe_values = jnp.array([1.0, 2.0, 3.0, 0.0])  # Assuming we treat 0.0 as Nothing
# final_state, outputs = scan_example(maybe_values)

# print("Final State:", final_state)
# print("Outputs:", outputs)


# import jax
# import jax.numpy as jnp
# from typing import Dict, Any
# import equinox as eqx


# class Test(eqx.Module):
#     tensor: jax.Array
#     static: str = eqx.field(static=True)


# # Define a function that returns a pytree with static fields
# def my_function(x):
#     return Test(tensor=x * 2, static="static_value")


# # Evaluate shape without computation
# shape_info = jax.eval_shape(my_function, jnp.ones((3, 3)))
# zero_initialized = jax.tree.map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), shape_info)
# print(zero_initialized.tensor)  # Should be (3, 3)


# # Define a new function g that transforms a PyTree
# def g(tree: Test):
#     return Test(tensor=tree.tensor + 1, static=tree.static)


# g_shape_info = jax.eval_shape(g, zero_initialized)
# zero_initialized = jax.tree.map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), g_shape_info)
# print(zero_initialized.tensor)  # Should be (3, 3)

# # # Define a function that maps one pytree to another
# # def map_pytree(pytree):
# #     return jax.tree_map(lambda x: x if isinstance(x, str) else x + 1, pytree)


# # # Test mapping function with eval_shape
# # mapped_shape_info = jax.eval_shape(map_pytree, shape_info)
# # print(jax.tree_map(lambda leaf: leaf if isinstance(leaf, str) else (leaf.shape, leaf.dtype), mapped_shape_info))

# import jax
# import jax.numpy as jnp
# import equinox as eqx


# class MyPyTree(eqx.Module):
#     data: jnp.ndarray  # Batched (N, D) array
#     condition: bool = eqx.field(static=True)  # Static field


# @eqx.filter_jit
# def use_lax_cond(pytree: MyPyTree) -> jnp.ndarray:
#     """Applies jax.lax.cond based on the PyTree's condition."""
#     return jax.vmap(
#         lambda x: jax.lax.cond(
#             pytree.condition,
#             lambda _: x**2,  # If condition=True → Square
#             lambda _: x + 1,  # If condition=False → Add 1
#             operand=None,
#         )
#     )(pytree.data)


# # Single PyTree with stacked data
# batched_pytree = MyPyTree(
#     data=jnp.array(
#         [
#             [1.0, 2.0, 3.0],  # Batch 1
#             [4.0, 5.0, 6.0],  # Batch 2
#         ]
#     ),
#     condition=False,  # Static field
# )

# # Apply function
# result = use_lax_cond(batched_pytree)

# print(result)


# import jax
# import jax.numpy as jnp
# import equinox as eqx


# class MyPyTree(eqx.Module):
#     data: jnp.ndarray  # (Batch, Features)
#     condition: bool = eqx.field(static=True)  # Static field


# @eqx.filter_jit
# def use_lax_cond(pytree: MyPyTree) -> MyPyTree:
#     """Applies jax.lax.cond based on the PyTree's condition and returns a new PyTree."""
#     new_data = jax.vmap(
#         lambda x: jax.lax.cond(
#             pytree.condition,
#             lambda _: x**2,  # If condition=True → Square
#             lambda _: x + 1,  # If condition=False → Add 1
#             operand=None,
#         )
#     )(pytree.data)

#     return MyPyTree(data=new_data, condition=pytree.condition)


# def step_fn(carry: MyPyTree, _):
#     """One scan step: applies transformation and returns updated PyTree."""
#     new_pytree = use_lax_cond(carry)
#     return new_pytree, new_pytree  # Carry and output both return PyTree


# # Initial state
# initial_pytree = MyPyTree(
#     data=jnp.array(
#         [
#             [1.0, 2.0, 3.0],  # Batch 1
#             [4.0, 5.0, 6.0],  # Batch 2
#         ]
#     ),
#     condition=True,  # Static field
# )

# # Scan over 3 steps
# T = 3
# final_pytree, pytree_history = jax.lax.scan(step_fn, initial_pytree, None, length=T)

# print("Final PyTree Data:\n", final_pytree.data)
# print("\nHistory of PyTrees:")
# for i, pytree in enumerate(pytree_history.data):
#     print(f"Step {i + 1}:\n", pytree)

# import jax
# import jax.numpy as jnp
# import equinox as eqx


# class MyPyTree(eqx.Module):
#     data: jnp.ndarray  # (Batch, Features)
#     condition: bool


# @eqx.filter_jit
# def use_lax_cond(pytree: MyPyTree) -> MyPyTree:
#     """Applies jax.lax.cond based on the PyTree's condition and returns a new PyTree."""
#     new_data = jax.vmap(
#         lambda x: jax.lax.cond(
#             pytree.condition,
#             lambda _: x**2,  # If condition=True → Square
#             lambda _: x + 1,  # If condition=False → Add 1
#             operand=None,
#         )
#     )(pytree.data)

#     return MyPyTree(data=new_data, condition=pytree.condition)


# def step_fn(carry: MyPyTree, step_index):
#     """Scan step: modifies condition at a specific step."""
#     new_pytree = use_lax_cond(carry)

#     # Change condition to False at step 2
#     new_condition = step_index < 1  # True for step 0, False for step 1 onward

#     updated_pytree = MyPyTree(data=new_pytree.data, condition=new_condition)
#     return updated_pytree, updated_pytree  # Carry & output both return PyTree


# # Initial state
# initial_pytree = MyPyTree(
#     data=jnp.array(
#         [
#             [1.0, 2.0, 3.0],  # Batch 1
#             [4.0, 5.0, 6.0],  # Batch 2
#         ]
#     ),
#     condition=True,  # Now a normal JAX boolean, not static
# )

# # Scan over 3 steps
# T = 3
# final_pytree, pytree_history = jax.lax.scan(step_fn, initial_pytree, jnp.arange(T))

# print(pytree_history.condition)

# print("Final PyTree Data:\n", final_pytree.data)
# print("\nHistory of PyTrees:")
# for i, pytree in enumerate(pytree_history.data):
#     print(f"Step {i + 1}:\n", pytree)


# import jax
# import jax.numpy as jnp
# import equinox as eqx


# class MyPyTree(eqx.Module):
#     data: jnp.ndarray  # (Batch, Features)
#     condition: jnp.ndarray


# # Define the scan function
# def scan_fn(carry, x):
#     arr, flag = carry.data, carry.condition
#     new_arr = arr + x  # Example operation
#     new_flag = False  # Change flag to False after first step
#     return MyPyTree(data=new_arr, condition=new_flag), new_arr


# # Inputs
# xs = jnp.array([1, 2, 3, 4])
# initial_carry = MyPyTree(data=jnp.array(0), condition=True)

# # Run scan
# final_carry, outputs = jax.lax.scan(scan_fn, initial_carry, xs)
# print(final_carry.condition)
# print("Final carry:", final_carry)
# print("Outputs:", outputs)


# import jax
# import jax.numpy as jnp
# import time


# # Define function using jax.lax.scan
# def scan_cumsum(xs):
#     def body(carry, x):
#         carry = carry + x
#         return carry, carry

#     _, result = jax.lax.scan(body, 0.0, xs)
#     return result


# # Define function using a jitted for-loop
# def loop_cumsum(xs):
#     carry = 0.0
#     result = []
#     for x in xs:
#         carry += x
#         result.append(carry)
#     return jnp.array(result)


# # Generate test data
# xs = jnp.arange(10_000, dtype=jnp.float32)

# # Measure compile time for scan
# t0 = time.time()
# scan_cumsum_jit = jax.jit(scan_cumsum)  # JIT compile
# scan_cumsum_jit(xs).block_until_ready()  # Trigger compilation
# t1 = time.time()
# scan_compile_time = t1 - t0

# # Measure execution time for scan
# t0 = time.time()
# scan_cumsum_jit(xs).block_until_ready()
# t1 = time.time()
# scan_exec_time = t1 - t0

# # Measure compile time for loop
# t0 = time.time()
# loop_cumsum_jit = jax.jit(loop_cumsum)  # JIT compile
# loop_cumsum_jit(xs).block_until_ready()  # Trigger compilation
# t1 = time.time()
# loop_compile_time = t1 - t0

# # Measure execution time for loop
# t0 = time.time()
# loop_cumsum_jit(xs).block_until_ready()
# t1 = time.time()
# loop_exec_time = t1 - t0

# # Print results
# print(f"JAX Scan Compile Time: {scan_compile_time:.6f} sec")
# print(f"JAX Scan Execution Time: {scan_exec_time:.6f} sec")
# print(f"Jitted Loop Compile Time: {loop_compile_time:.6f} sec")
# print(f"Jitted Loop Execution Time: {loop_exec_time:.6f} sec")


# import jax
# import jax.numpy as jnp


# def scan_fn(carry, x):
#     new_carry = lambda y: carry(x(y))  # Chain the functions
#     return new_carry, None


# # Initial carry is the identity function
# init_carry = lambda x: x

# # Inputs to the scan (could be any functions)
# inputs = [lambda x: x + 1, lambda x: x * 2, lambda x: x - 3]

# final_carry, _ = jax.lax.scan(scan_fn, init_carry, inputs)

# # Test the resulting function
# x = 5
# print(final_carry(x))  # Should compute ((5 + 1) * 2) - 3


# from dataclasses import dataclass


# class Test[A]:
#     type test = list[A]

#     def __init__(self, tttt: test):
#         self.tttt = tttt


# a = Test[int]([1, 2, 3])


# import jax
# import jax.numpy as jnp
# import equinox as eqx


# # Define the PyTree as an Equinox Module
# class MyPyTree(eqx.Module):
#     condition: bool
#     value: jax.Array


# # Define a function operating on the PyTree
# def my_function(pytree: MyPyTree):
#     return jnp.sum(pytree.value**2) if pytree.condition else jnp.sum(pytree.value)


# # Compute the gradient w.r.t. the PyTree value
# grad_fn = eqx.filter_jacrev(my_function)

# # Create an instance of MyPyTree
# pytree_instance = MyPyTree(condition=True, value=jnp.array([1.0, 2.0, 3.0]))

# # Compute and print the gradient
# grad_result = grad_fn(pytree_instance)
# print(grad_result)


# import jax
# import jax.numpy as jnp
# import equinox as eqx
# import jax.lax as lax


# # Define the PyTree as an Equinox Module
# class MyPyTree(eqx.Module):
#     condition: bool
#     value: jax.Array


# # Define a function operating on the PyTree
# def my_function(pytree: MyPyTree) -> MyPyTree:
#     condition = pytree.condition
#     new_value = lax.cond(
#         condition,
#         lambda _: pytree.value**2,  # If True, square the values
#         lambda _: pytree.value + 1,  # If False, add 1 to values
#         operand=None,
#     )
#     return new_value, condition


# # Compute the gradient w.r.t. the PyTree value
# grad_fn = eqx.filter_jacrev(my_function, has_aux=True)

# # Create an instance of MyPyTree
# pytree_instance = MyPyTree(condition=True, value=jnp.array([1.0, 2.0, 3.0]))

# # Compute and print the gradient
# grad_result, condition = grad_fn(pytree_instance)
# print(grad_result)
# print(condition)


# import jax
# import jax.numpy as jnp


# def my_function(x: jnp.ndarray, cls: type) -> jnp.ndarray:
#     """A JIT-compiled function that takes a static class type but doesn't use it."""
#     return x * 2  # cls is not used


# # JIT compile with `cls` as a static argument
# jit_func = jax.jit(my_function, static_argnames=["cls"])

# # Example usage
# import numpy as np


# class Dummy:
#     pass


# x = jnp.array([1, 2, 3])
# result = jit_func(x, Dummy)  # Pass the class type
# print(result)  # Output: [2 4 6]


# import jax
# import jax.numpy as jnp
# import optax


# # Define a simple loss function
# def loss_fn(params):
#     return jnp.sum(params**2)  # Example: L2 loss


# # Initialize SGD optimizer
# learning_rate = 0.1
# optimizer = optax.sgd(learning_rate=learning_rate)
# print(optimizer)


# # Example parameters
# params = jnp.array([1.0, -2.0, 3.0])

# # Initialize optimizer state
# opt_state = optimizer.init(params)

# # Compute gradients
# grad_fn = jax.grad(loss_fn)
# grads = grad_fn(params)

# # Compute updates
# updates, new_opt_state = optimizer.update(grads, opt_state, params)

# # Apply updates to parameters
# new_params = optax.apply_updates(params, updates)

# # Print everything
# print(opt_state)
# print("Params before update: ", params)
# print("Gradients: ", grads)
# print("Updates: ", updates)
# print("Params after update: ", new_params)
# print("Opt State: ", new_opt_state)


# import jax
# import jax.numpy as jnp
# import optax
# from typing import Any, NamedTuple
# import equinox as eqx


# # Define a PyTree structure
# class MyState(NamedTuple):
#     optimizer: Any  # Placeholder for Optax optimizer
#     value: jnp.ndarray  # Some arbitrary state


# # Define an initial state with SGD optimizer
# init_optimizer = optax.sgd(learning_rate=0.1)
# initial_state = MyState(optimizer=init_optimizer, value=jnp.array([1.0, 2.0, 3.0]))


# # Function that modifies the optimizer inside the PyTree
# def update_fn(state: MyState):
#     new_optimizer = optax.adam(learning_rate=0.01)  # Change optimizer to Adam
#     new_state = state._replace(optimizer=new_optimizer)
#     return jnp.sum(state.value), new_state  # Returning some computed output and the modified state


# # JIT compile the function
# jit_update_fn = eqx.filter_jit(update_fn)

# # Apply the function
# output, new_state = jit_update_fn(initial_state)

# print("Output:", output)
# print("New Optimizer Type:", type(new_state.optimizer))


# import jax
# import jax.numpy as jnp
# import optax
# import equinox as eqx

# jax.config.update("jax_enable_x64", True)


# # Define a PyTree structure that stores only the learning rate and a trainable parameter
# class MyState(eqx.Module):
#     learning_rate: float
#     params: jnp.ndarray  # Trainable parameters


# # Initialize state with a learning rate and parameters
# initial_state = MyState(learning_rate=0.1, params=jnp.array([1.0, 2.0, 3.0]))


# # Define a loss function
# def loss_fn(params):
#     return jnp.sum(params**2)  # Simple quadratic loss


# # Function that updates parameters using dynamically created optimizer
# def update_fn(state: MyState):
#     # Dynamically create optimizer
#     optimizer = optax.sgd(learning_rate=state.learning_rate)

#     # Compute gradient of loss w.r.t. parameters
#     grads = jax.grad(loss_fn)(state.params)

#     # Initialize optimizer state (momentum, accumulators, etc.)
#     opt_state = optimizer.init(state.params)

#     # Apply the optimizer update
#     updates, opt_state = optimizer.update(grads, opt_state, state.params)
#     new_params = optax.apply_updates(state.params, updates)

#     # Return updated parameters and modified state
#     new_state = MyState(learning_rate=state.learning_rate, params=new_params)
#     return jnp.sum(new_params), new_state  # Returning a computed value and updated state


# # JIT compile the function
# jit_update_fn = eqx.filter_jit(update_fn)  # <--- Now we're actually going to use this

# # **Apply the function using JIT**
# output, new_state = jit_update_fn(initial_state)

# print("JIT Output:", output)
# print("Updated Parameters:", new_state.params)

# # Compute the autodiff gradient w.r.t. learning rate
# grad_fn = jax.grad(lambda s: jit_update_fn(s)[0])  # Use the JIT function for gradients
# auto_grad = grad_fn(initial_state)

# # Numerical gradient check
# epsilon = 1e-4
# state_plus = MyState(learning_rate=initial_state.learning_rate + epsilon, params=initial_state.params)
# state_minus = MyState(learning_rate=initial_state.learning_rate - epsilon, params=initial_state.params)

# numerical_grad = (jit_update_fn(state_plus)[0] - jit_update_fn(state_minus)[0]) / (2 * epsilon)

# # Print results
# print("Autodiff Gradient:", auto_grad.learning_rate)
# print("Numerical Gradient:", numerical_grad)
# print("Gradient Check Passed:", jnp.allclose(auto_grad.learning_rate, numerical_grad, atol=1e-5))


# from dataclasses import dataclass
# import wandb


# # Step 1: Define your configuration class using dataclass
# @dataclass
# class MyConfig:
#     learning_rate: float
#     batch_size: int
#     model_name: str
#     use_augmentation: bool


# # Step 2: Initialize a new wandb run with the configuration
# wandb.init(config={"learning_rate": 0.001, "batch_size": 32, "model_name": "resnet", "use_augmentation": True})

# # Step 3: Map wandb.config to your typed config class
# config = MyConfig(**wandb.config)

# # Step 4: Test by printing out the config
# print("WandB Config:", wandb.config)  # Prints the raw wandb.config
# print("Typed Config:", config)  # Prints the typed config object

# # Additional: Check individual attributes to make sure they match
# assert config.learning_rate == 0.001, f"Expected 0.001, but got {config.learning_rate}"
# assert config.batch_size == 32, f"Expected 32, but got {config.batch_size}"
# assert config.model_name == "resnet", f"Expected 'resnet', but got {config.model_name}"
# assert config.use_augmentation is True, f"Expected True, but got {config.use_augmentation}"

# print("All assertions passed! Configuration is properly mapped.")


# import jax
# import jax.numpy as jnp
# from jax.flatten_util import ravel_pytree

# # Define a pytree (nested structure)
# tree = {"a": jnp.array([1.0, 2.0]), "b": (jnp.array([3.0]), jnp.array([4.0, 5.0]))}

# # Flatten the pytree
# flat_array, unravel_fn = ravel_pytree(tree)

# print("Flattened array:", flat_array)

# # Attempt to unflatten with a wrong-sized array
# wrong_size_array = jnp.array([33.0, 2.0, 3.0, 4.0, 5.0])  # Incorrect size
# try:
#     new_tree = unravel_fn(wrong_size_array)
# except ValueError as e:
#     print("Error:", e)

# print(new_tree)


# import jax
# import jax.numpy as jnp


# # Define a JIT-compiled function
# # @jax.jit
# def my_function(x):
#     return x * 2


# # result = my_function(4)


# print(jax.eval_shape(my_function, None))


# import jax
# import jax.numpy as jnp
# import optax


# # Define a simple loss function
# def loss_fn(params):
#     return jnp.sum(params**2)


# # Initialize parameters
# params = jnp.array([1.0, -2.0, 3.0])

# # Create Adam optimizer
# optimizer = optax.adam(learning_rate=0.1)

# # Normally, we'd initialize state with optimizer.init(params)
# # Instead, we use an empty state
# opt_state = optax.EmptyState()  # Not recommended for Adam

# # Compute gradients
# grads = jax.grad(loss_fn)(params)

# # Attempt to apply the optimizer update
# try:
#     updates, opt_state = optimizer.update(grads, opt_state, params)
#     params = optax.apply_updates(params, updates)
#     print("Updated params:", params)
# except Exception as e:
#     print("Error:", e)


# import jax
# import jax.numpy as jnp


# # Define some functions that transform the accumulator
# def f1(a):
#     return a * 2


# def f2(a):
#     return a + 3


# def f3(a):
#     return a**2


# # Hardcoded static list of functions
# fs = [f1, f2, f3]


# # Function applying each function in sequence
# def apply_functions(a):
#     for f in fs:  # Just a normal Python loop!
#         a = f(a)
#     return a


# # JIT compile
# jit_apply_functions = jax.jit(apply_functions)

# # Test
# print(jit_apply_functions(2.0))  # Should output 49 ((2 * 2) + 3) ** 2
# print(jit_apply_functions(2.0))  # Should output 49 ((2 * 2) + 3) ** 2
# print(jit_apply_functions(2.0))  # Should output 49 ((2 * 2) + 3) ** 2
# print(jit_apply_functions(2.0))  # Should output 49 ((2 * 2) + 3) ** 2
# print(jit_apply_functions(2.0))  # Should output 49 ((2 * 2) + 3) ** 2

# import jax
# import jax.numpy as jnp
# import equinox as eqx


# # Define a PyTree with None values
# class MyTree(eqx.Module):
#     param1: jnp.ndarray | None
#     param2: jnp.ndarray | None


# tree = MyTree(param1=None, param2=None)

# # Define actual arrays to update the tree
# new_values = MyTree(param1=jnp.array([1.0, 2.0, 3.0]), param2=jnp.array([[4.0, 5.0], [6.0, 7.0]]))

# # Update param1 first
# updated_tree = eqx.tree_at(lambda t: t.param1, tree, new_values.param1, is_leaf=lambda x: x is None)

# # Update param2 next
# updated_tree = eqx.tree_at(lambda t: t.param2, updated_tree, new_values.param2, is_leaf=lambda x: x is None)

# print(updated_tree)


import jax
import jax.numpy as jnp
import optax


def normalized_sgd(learning_rate):
    return optax.chain(
        optax.normalize_by_update_norm(scale_factor=1.0),  # Normalize the gradient
        optax.sgd(learning_rate),  # Apply learning rate scaling
    )


# Quick test
def test_optimizer():
    def loss_fn(x):
        return jnp.sum(x**2)  # Simple quadratic loss

    params = jnp.array([3.0, 4.0])  # Initial parameters
    optimizer = normalized_sgd(learning_rate=0.1)
    opt_state = optimizer.init(params)

    for i in range(5):
        grads = jax.grad(loss_fn)(params)  # Compute gradients
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        print(f"Step {i + 1}, Params: {params}, Loss: {loss_fn(params):.6f}")


test_optimizer()
