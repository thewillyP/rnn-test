from typing import Generator, NewType
import jax
import jax.numpy as jnp
import equinox as eqx

type G[T] = Generator[None, None, T]


ACTIVATION = NewType("ACTIVATION", jax.Array)
PREDICTION = NewType("PREDICTION", jax.Array)
PARAMETER = NewType("PARAMETER", jax.Array)
LOSS = NewType("LOSS", jnp.float32)
LEARNING_RATE = NewType("LEARNING_RATE", jnp.float32)
PRNG = NewType("PRNG", jax.Array)


# bc grad over param pytree returns a param pytree, and need a way to distinguish between gr vs param, but still recognize they're same type
class Gradient[T: eqx.Module](eqx.Module):
    value: T


# smart constructor to let me know that some PyTrees are meant to be iterated
class Traversable[T: eqx.Module](eqx.Module):
    value: T | jax.Array
