import itertools
import torch 

torch.manual_seed(0)

# sample uniform between [0.005, 0.06] learning rate
learning_rates = torch.rand(10) * 0.055 + 0.005

# Parameter grid
params = {
    "n_in": [2],
    "n_h": [30],
    "n_out": [1],
    "num_layers": [1],
    "task": ["Random"],
    "seq": [100],
    "numVl": [2],
    "numTe": [5000],
    "numTr": [4], 
    "batch_size_tr": [2],
    "batch_size_vl": [4],
    "batch_size_te": [5000],
    "num_epochs": [1600], 
    "learning_rate": learning_rates,
    # "learning_rate": [0.1, 0.01, 0.3],
    "optimizerFn": ["SGD"],
    "lossFn": ["mse"],
    "mode": ["experiment"],
    "checkpoint_freq": [800],
    "seed": list(range(3, 3+1)),
    "projectName": ["oho-rnn-generalization-gap"],
    "logger": ["wandb"],
    "performance_samples": [9],
    "init_scheme": ['StaticRandomInit'],
    "activation_fn": ['tanh'],
    "log_freq": [1],
    "meta_learning_rate": [0.0005],
    "l2_regularization": [0],
    "is_oho": [1, 0],
    "randomType": ["Uniform"],
    "time_chunk_size": [10],
    "t1": [3],
    "t2": [5],
}

# Generate all combinations
keys, values = zip(*params.items())
other_combinations = itertools.product(*values)

# Final combinations
final_combinations = []
for combo in other_combinations:
    base_combo = dict(zip(keys, combo))
    final_combinations.append(base_combo)
    # for train_batch in train_batch_combinations:
    #     for t_combo in t_combinations:
    #         new_combo = base_combo.copy()
    #         new_combo.update(train_batch)
    #         new_combo.update(t_combo)
    #         final_combinations.append(new_combo)

# Write to configs.txt
with open("configs.txt", "w") as file:
    for combo in final_combinations:
        args = " ".join([f"--{key} {value}" for key, value in combo.items()])
        file.write(args + "\n")

print(f"Generated {len(final_combinations)} configurations in configs.txt")
