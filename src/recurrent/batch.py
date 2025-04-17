from typing import Any
from recurrent.myrecords import *
from recurrent.myrecords import GodState


def batched_values(env: GodState, replace_fn: Callable[[Any], Any]) -> GodState:
    def get_batched(env: GodState):
        return (
            env.prng,
            env.rnnState.activation,
            env.innerInfluenceTensor,
            env.innerUoro,
            env.innerLogs.influenceTensor,
            env.innerLogs.immediateInfluenceTensor,
            env.innerLogs.jac_eigenvalue,
            env.innerLogs.hessian,
        )

    batched_env = jax.tree.map(lambda _: None, env)
    batched_env = eqx.tree_at(
        get_batched,
        batched_env,
        replace=tuple(replace_fn(_) for _ in get_batched(env)),
        is_leaf=lambda x: x is None,
    )
    return batched_env


def batch_env_form(env: GodState) -> GodState:
    return batched_values(env, lambda _: 0)


def keep_only_batched_values(env: GodState) -> GodState:
    return batched_values(env, lambda x: x)


def to_batched_form(env: GodState, env_unbatched: GodState) -> GodState:
    _env: GodState = keep_only_batched_values(env)
    return eqx.combine(_env, env_unbatched)


# single env -> batch template -> input to batched env
