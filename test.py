import random
import json


def generate_seeds(num_entries=3, test_seed=12345):
    seeds = []
    for _ in range(num_entries):
        entry = {
            "data_seed": random.randint(0, 99999),
            "parameter_seed": random.randint(0, 99999),
            "test_seed": test_seed,
        }
        seeds.append(entry)
    return seeds


# Example usage
if __name__ == "__main__":
    seed_list = generate_seeds(num_entries=10)
    print(json.dumps(seed_list, indent=2))
