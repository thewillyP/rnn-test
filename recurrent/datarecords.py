import jax
import equinox as eqx
from recurrent.mytypes import *


class InputOutput(eqx.Module):
    x: jax.Array
    y: jax.Array


class OhoInputOutput(eqx.Module):
    train: Traversable[InputOutput]
    validation: Traversable[InputOutput]
    labels: Traversable[jax.Array]
