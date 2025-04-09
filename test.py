# test.py
from sweep_agent.agent import get_sweep_config


def main():
    sweep_config = get_sweep_config()
    if sweep_config:
        print(f"Running sweep {sweep_config.sweep_id}")
        print(f"Program: {sweep_config.program}")
        print(f"Experiment name: {sweep_config.name}")
        print(f"Hyperparameters: {sweep_config.config}")
        # Your training logic here using sweep_config.config for hyperparameters
    else:
        print("No sweep config received")


if __name__ == "__main__":
    main()
