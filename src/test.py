from dacite import Config, from_dict
from sweep_agent.agent import get_sweep_config
from recurrent.app import create_datasets, runApp, create_env
from recurrent.batch import *


def main():
    sweep_config = get_sweep_config()
    if not sweep_config:
        raise ValueError("No sweep config received")

    config = from_dict(
        data_class=GodConfig, data=sweep_config.config, config=Config(type_hooks={tuple[int, int]: tuple})
    )

    prng = PRNG(jax.random.key(config.seed.parameter_seed))
    prng1, prng2 = jax.random.split(prng, 2)
    prngs = jax.random.split(prng2, 2)

    _env = create_env(config, prng1)[0]

    print(_env)
    print(prngs)

    create_envs = eqx.filter_vmap(lambda c, p: create_env(c, p)[0], in_axes=(None, 0))
    create_envs = eqx.filter_jit(create_envs)

    env = create_envs(config, prngs)

    env = to_batched_form(env, _env)

    print(env)

    create_ds = eqx.filter_vmap(lambda c, p: create_datasets(c, p, prng), in_axes=(None, 0), out_axes=(0, None))
    ds, ds_test = create_ds(config, prngs)
    print(ds)
    print(ds_test)


if __name__ == "__main__":
    main()
