from sweep_agent.agent import get_sweep_config
from recurrent.app import runApp, create_env

# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)


def main():
    sweep_config = get_sweep_config()
    if not sweep_config:
        raise ValueError("No sweep config received")

    load_env = lambda config, prng: create_env(config, prng)[0]
    load_config = lambda run: run.config
    wandb_kwargs = {"mode": "offline", "group": sweep_config.name, "config": sweep_config.config, "project": "oho_exps"}

    runApp(load_env, load_config, wandb_kwargs)


if __name__ == "__main__":
    main()
