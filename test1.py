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


import time
import torch


def create_recursive_closure(tensor, n):
    def closure(tensor, remaining):
        if remaining == 0:
            return tensor
        else:
            # Example operation: multiplying tensor by 2 (you can replace this with your desired operation)
            result = tensor * 2
            return create_recursive_closure(result, remaining - 1)

    return closure(tensor, n)


# Example usage:
input_tensor = torch.tensor([1.0, 2.0, 3.0])
n = 5
model = torch.compile(create_recursive_closure)
start = time.time()
output_tensor = model(input_tensor, n)
print(time.time() - start)
start = time.time()
output_tensor = model(input_tensor, n)
print(time.time() - start)
start = time.time()
output_tensor = model(input_tensor, n)
print(time.time() - start)
start = time.time()
output_tensor = model(input_tensor, n)
print(time.time() - start)
start = time.time()
output_tensor = model(input_tensor, n)
print(time.time() - start)
# print(output_tensor)
