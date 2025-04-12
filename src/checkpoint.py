import wandb
import copy
import jax
import os
import sys
from dataclasses import dataclass
from recurrent.app import runApp, load_artifact
from recurrent.myrecords import GodState


# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)


@dataclass
class CommandLine:
    run_id: str
    entity: str
    project: str
    version: str


def parse_args():
    if len(sys.argv) == 4:
        return CommandLine(
            run_id=sys.argv[1], entity="wlp9800-new-york-university", project=sys.argv[2], version=sys.argv[3]
        )
    else:
        raise ValueError("When running manually, please provide run_id and project as arguments")


def main(cmd: CommandLine):
    artifact_name = f"{cmd.entity}/{cmd.project}/trained_env_{cmd.run_id}:{cmd.version}"
    env_artifact: GodState = load_artifact(artifact_name, "env.pkl")
    env_artifact = copy.replace(env_artifact, prng=jax.random.wrap_key_data(env_artifact.prng))
    load_env = lambda _, __: env_artifact

    api = wandb.Api()
    original_run = api.run(f"{cmd.entity}/{cmd.project}/{cmd.run_id}")
    original_config = original_run.config
    load_config = lambda _: original_config
    wandb_kwargs = {"mode": "offline", "config": original_config}

    runApp(load_env, load_config, wandb_kwargs=wandb_kwargs)


if __name__ == "__main__":
    cmd = parse_args()
    main(cmd)
