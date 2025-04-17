# import jax
# import jax.numpy as jnp
# import equinox as eqx


# class NestedPytree(eqx.Module):
#     static_field: float = eqx.field(static=True)  # Static field inside nested pytree
#     batched_field: jnp.ndarray  # Batched field inside nested pytree


# class MyInput(eqx.Module):
#     a: jnp.ndarray
#     b: jnp.ndarray
#     scale: float = eqx.field(static=True)  # Static field
#     mode: int  # Dynamic (but not batched)
#     nested: NestedPytree  # A nested pytree


# def my_func(example: MyInput):
#     weighted_sum = jnp.sum(example.a * example.b)
#     score = example.scale * weighted_sum if example.mode == 0 else weighted_sum + example.scale

#     # Creating a new nested pytree instance
#     new_nested = NestedPytree(
#         static_field=example.nested.static_field,  # Static field stays the same
#         batched_field=example.nested.batched_field + 1.0,  # Batched field is modified
#     )

#     # Return a scalar and a new MyInput instance with the updated nested pytree
#     new_out = MyInput(a=example.a + 1.0, b=example.b + 1.0, scale=example.scale, mode=example.mode, nested=new_nested)

#     return score, new_out  # Output is (scalar, MyInput)


# # Create batched input with nested pytree
# batched_input = MyInput(
#     a=jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
#     b=jnp.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
#     scale=0.1,
#     mode=0,
#     nested=NestedPytree(
#         static_field=0.5,  # Static field
#         batched_field=jnp.array([1.0, 2.0, 3.0]),  # Batched field
#     ),
# )

# # Static field must match exactly
# in_axes = jax.tree.map(lambda _: None, batched_input)
# in_axes = eqx.tree_at(
#     lambda x: (x.a, x.b, x.nested.batched_field), in_axes, replace=(0, 0, 0), is_leaf=lambda x: x is None
# )
# print(in_axes)

# # in_axes = MyInput(a=0, b=0, scale=batched_input.scale, mode=None, nested=0)

# # filter_vmap with correct structure
# vmapped_func = eqx.filter_vmap(my_func, in_axes=(in_axes,))

# # Call with a tuple
# result = vmapped_func(batched_input)

# # Unpack
# scores, out_modules = result

# print("Scores:", scores)
# print("Out modules:")
# print("  a:", out_modules.a)
# print("  b:", out_modules.b)
# print("  scale:", out_modules.scale)
# print("  mode:", out_modules.mode)
# print("  Nested pytree:")
# print("    static_field:", out_modules.nested.static_field)
# print("    batched_field:", out_modules.nested.batched_field)


# import jax
# import jax.numpy as jnp


# def f(x):
#     return jnp.stack([x, x])  # returns shape (2,)


# batched = jax.vmap(f, out_axes=None)
# print(batched(jnp.array([1, 2, 3])))


import jax
import jax.numpy as jnp


@jax.jit
def double_assign(x):
    x = x + 1
    x = x * 2
    return x


print(double_assign(jnp.array([1, 2, 3])))
