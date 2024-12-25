import torch
from torch.func import jacrev

# Define a linear function f(x) = W @ x
def my_function(x):
    W = torch.randn(4, 3)  # Weight matrix of shape (4, 3)
    return W @ x

# Input vector (shape 3)
x = torch.randn(3, requires_grad=True)

# Compute the Jacobian matrix
jacobian = jacrev(my_function)(x)

print("Jacobian shape:", jacobian.shape)  # Shape will be (4, 3)
print(jacobian)  # The Jacobian matrix