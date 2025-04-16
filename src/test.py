from sweep_agent.agent import get_sweep_config
from recurrent.app import runApp, create_env
from recurrent.batch import *


def main():
    sweep_config = get_sweep_config()
    if not sweep_config:
        raise ValueError("No sweep config received")

    config = GodConfig(**sweep_config.config)
    prng = PRNG(jax.random.key(config.seed))
    prng1, prng2 = jax.random.split(prng, 2)
    prngs = jax.random.split(prng2, 2)

    _env = create_env(config, prng1)[0]

    print(_env)
    print(prngs)

    create_envs = eqx.filter_vmap(lambda c, p: create_env(c, p)[0], in_axes=(None, 0))

    env = create_envs(config, prngs)

    env = to_batched_form(env, _env)

    print(env)


if __name__ == "__main__":
    main()
