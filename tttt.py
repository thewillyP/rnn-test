from typing import Protocol, TypeVar

# Define the Protocols
class RNN_LEARNABLE(Protocol):
    def train(self) -> None:
        """Train the model."""
        pass

class WithBaseFuture(Protocol):
    def predict_future(self) -> str:
        """Predict future outcomes."""
        pass

# Combine into a single Protocol-like class
class _BaseFutureCap(RNN_LEARNABLE, WithBaseFuture, Protocol):
    pass

# Define a TypeVar bound to the combined Protocol
T = TypeVar("T", bound=_BaseFutureCap)

# Function that uses the TypeVar
def use_future_cap(obj: T) -> None:
    obj.train()
    # print(obj.predict_future())

# Dummy class that satisfies the Protocol
class MyModel:
    def train(self) -> None:
        print("Training the model...")

    def predict_future(self) -> str:
        return "The future looks bright!"

# Dummy class that does NOT satisfy the Protocol
class IncompleteModel:
    def train(self) -> None:
        print("Training the incomplete model...")

# Main function to test
if __name__ == "__main__":
    model = MyModel()
    use_future_cap(model)  # Works fine

    incomplete_model = IncompleteModel()
    # Uncommenting the line below will cause a mypy error
    # use_future_cap(incomplete_model)