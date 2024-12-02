import itertools

# Parameter grid
params = {
    "n_in": [2],
    "n_h": [30, 200],
    "n_out": [1],
    "num_layers": [1],
    "task": ["Random"],
    "seq": [10, 20],
    "numVl": [1],
    "numTe": [5000],
    "batch_size_vl": [1],
    "batch_size_te": [1000],
    "num_epochs": [35000],
    "learning_rate": [0.001, 0.01, 0.1],
    "optimizerFn": ["SGD"],
    "lossFn": ["mse"],
    "mode": ["experiment"],
    "checkpoint_freq": [700],
    "seed": list(range(1, 10+1)),
    "projectName": ["rnn-test-sgdspike"],
    "logger": ["wandb"],
    "performance_samples": [9],
}

# Pairing for numTr and batch_size_tr
train_batch_combinations = [
    {"numTr": 1000, "batch_size_tr": 1000},
]

# Convert t_combinations to dictionary style
t_combinations = [
    {"t1": 1, "t2": 1},
    {"t1": 2, "t2": 2},
    {"t1": 3, "t2": 5},
    {"t1": 5, "t2": 9}
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
