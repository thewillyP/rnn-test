import sys
import yaml
from sweep_agent.agent import get_sweep_config
from recurrent.app import runApp, create_env


def extract_first_config(yaml_dict):
    config = {}
    params = yaml_dict.get("parameters", {})
    for key, val in params.items():
        if "value" in val:
            config[key] = val["value"]
        elif "values" in val:
            config[key] = val["values"][0]
    return config


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/main.py <config_file.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    config = extract_first_config(yaml_config)
    load_env = lambda config, prng: create_env(config, prng)[0]
    load_config = lambda run: run.config
    wandb_kwargs = {"mode": "offline", "group": "test", "config": config, "project": "oho_exps"}

    runApp(load_env, load_config, wandb_kwargs)


if __name__ == "__main__":
    main()
