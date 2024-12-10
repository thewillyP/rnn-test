import itertools

# Parameter grid
params = {
    "n_in": [2],
    "n_h": [200, 30],
    "n_out": [1],
    "num_layers": [1],
    "task": ["Random"],
    "seq": [100],
    "numVl": [500],
    "numTe": [5000],
    "batch_size_vl": [500],
    "batch_size_te": [5000],
    "num_epochs": [10000], 
    "learning_rate": [0.1, 0.01, 0.3],
    "optimizerFn": ["SGD"],
    "lossFn": ["mse"],
    "mode": ["experiment"],
    "checkpoint_freq": [2000],
    "seed": list(range(1, 2+1)),
    "projectName": ["rnn-test-ohotest3"],
    "logger": ["wandb"],
    "performance_samples": [9],
    "init_scheme": ['ZeroInit'],
    "activation_fn": ['relu', 'tanh'],
    "log_freq": [2],
    "meta_learning_rate": [1e-4],
    "l2_regularization": [0],
    "is_oho": [1],
    "randomType": ["Uniform"],
    "time_chunk_size": [10]
}

# Pairing for numTr and batch_size_tr
train_batch_combinations = [
    {"numTr": 500, "batch_size_tr": 500},
    {"numTr": 500, "batch_size_tr": 100},
]

# Convert t_combinations to dictionary style
t_combinations = [
    {"t1": 3, "t2": 5},
    {"t1": 9, "t2": 1},
]

# Generate all combinations
keys, values = zip(*params.items())
other_combinations = itertools.product(*values)

# Final combinations
final_combinations = []
for combo in other_combinations:
    base_combo = dict(zip(keys, combo))
    for train_batch in train_batch_combinations:
        for t_combo in t_combinations:
            new_combo = base_combo.copy()
            new_combo.update(train_batch)
            new_combo.update(t_combo)
            final_combinations.append(new_combo)

# Write to configs.txt
with open("configs.txt", "w") as file:
    for combo in final_combinations:
        args = " ".join([f"--{key} {value}" for key, value in combo.items()])
        file.write(args + "\n")

print(f"Generated {len(final_combinations)} configurations in configs.txt")
