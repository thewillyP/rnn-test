import jax
import equinox as eqx


class InputOutput(eqx.Module):
    x: jax.Array
    y: jax.Array
