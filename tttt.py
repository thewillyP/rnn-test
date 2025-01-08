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
import torch
from torch.utils import _pytree as pytree
from dataclasses import dataclass


@dataclass(frozen=True)
class CustomData:
    value: torch.Tensor  # Batched tensor
    aux: int  # Static metadata (not batched)


# Register the PyTree
def customdata_flatten(custom_data: CustomData):
    return (custom_data.value,), custom_data.aux


def customdata_unflatten(children, aux):
    return CustomData(value=children[0], aux=aux)


pytree.register_pytree_node(CustomData, customdata_flatten, customdata_unflatten)


def tester(data: CustomData):
    return CustomData(data.value, data.aux + 10)


def process_custom_data(data: CustomData):
    # Simulate a more expensive computation
    env = CustomData(
        data.value**data.aux, data.aux + 5
    )  # also works!: data.aux + torch.ceil(data.value.mean()).int()
    return tester(env)


# Batched input
values = torch.randn(100000, 1)  # Larger batched data for meaningful benchmarking
aux_value = 2  # Static metadata (not batched)

batched_data = CustomData(value=values, aux=aux_value)


# Benchmark manual loop
def manual_loop(data):
    results = []
    for i in range(data.value.shape[0]):
        single_data = CustomData(data.value[i], data.aux)
        results.append(process_custom_data(single_data))
    return CustomData(value=torch.stack([r.value for r in results]), aux=results[0].aux)


# Measure vmap execution time
start_time_vmap = time.time()
result_vmap = torch.vmap(process_custom_data)(batched_data)
vmap_time = time.time() - start_time_vmap

# # Measure manual loop execution time
# start_time_loop = time.time()
# result_loop = manual_loop(batched_data)
# loop_time = time.time() - start_time_loop

# # Validate correctness
# assert torch.allclose(result_vmap.value, result_loop.value)
# assert result_vmap.aux == result_loop.aux

print(f"vmap execution time: {vmap_time:.6f} seconds")
# print(f"Manual loop execution time: {loop_time:.6f} seconds")

print(result_vmap)
