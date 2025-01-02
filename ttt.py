# import torch
# from torch.func import jacrev

# # Define a linear function f(x) = W @ x
# def my_function(x):
#     W = torch.randn(4, 3)  # Weight matrix of shape (4, 3)
#     return W @ x

# # Input vector (shape 3)
# x = torch.randn(3, requires_grad=True)

# # Compute the Jacobian matrix
# jacobian = jacrev(my_function)(x)

# print("Jacobian shape:", jacobian.shape)  # Shape will be (4, 3)
# print(jacobian)  # The Jacobian matrix


import torch
from torch.func import jacrev
from dataclasses import dataclass, replace

@dataclass(frozen=True)
class InputData:
    x: torch.Tensor
    y: str

def compute_loss(x: torch.Tensor) -> torch.Tensor:
    data = InputData(x=x, y="hi")
    new_data = replace(data, y="hello")
    new_data = replace(new_data, x=new_data.x * 4)
    loss = (new_data.x ** 2).sum()
    return loss

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

jacobian_fn = jacrev(compute_loss)
jacobian = jacobian_fn(x)

print("Loss:", compute_loss(x).item())
print("Jacobian wrt x:", jacobian)


# Compute the loss
loss = compute_loss(x)

grad = torch.autograd.grad(loss, x)[0]

# Print the computed loss and the gradient with respect to `x`
print("Autograd Loss:", loss.item())
print("Autograd Gradient wrt x:", grad)