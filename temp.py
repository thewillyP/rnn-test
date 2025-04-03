# # # # # # # import torch
# # # # # # # from torch.utils.data import Dataset, DataLoader
# # # # # # # import jax
# # # # # # # import jax.numpy as jnp
# # # # # # # import numpy as np

# # # # # # # # Example PyTree dataset (could be more nested)
# # # # # # # dataset = {
# # # # # # #     "features": jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=jnp.float32),
# # # # # # #     "labels": jnp.array([0, 1, 0, 1], dtype=jnp.float32),
# # # # # # #     "metadata": {"id": jnp.array([10, 20, 30, 40], dtype=jnp.int32)},
# # # # # # # }


# # # # # # # class PyTreeDataset(Dataset):
# # # # # # #     def __init__(self, pytree_data):
# # # # # # #         self.data = pytree_data
# # # # # # #         leaves = jax.tree.leaves(pytree_data)
# # # # # # #         if not leaves:
# # # # # # #             raise ValueError("PyTree has no leaves!")
# # # # # # #         self.n_samples = len(leaves[0])  # Assume all leaves have same length

# # # # # # #     def __len__(self):
# # # # # # #         return self.n_samples

# # # # # # #     def __getitem__(self, idx):
# # # # # # #         # Return a PyTree for the given index
# # # # # # #         sample = jax.tree.map(lambda x: x[idx], self.data)
# # # # # # #         return sample  # PyTree with JAX arrays


# # # # # # # # Custom collate function to convert JAX arrays to PyTorch tensors and batch them
# # # # # # # def pytree_collate_fn(batch):
# # # # # # #     # Batch is a list of PyTree samples; stack them into a batched PyTree
# # # # # # #     batched = jax.tree.map(lambda *xs: torch.stack([torch.from_numpy(np.array(x)) for x in xs]), *batch)
# # # # # # #     return batched


# # # # # # # # Create DataLoader
# # # # # # # pytorch_dataset = PyTreeDataset(dataset)
# # # # # # # dataloader = DataLoader(
# # # # # # #     pytorch_dataset,
# # # # # # #     batch_size=2,
# # # # # # #     shuffle=False,
# # # # # # #     collate_fn=pytree_collate_fn,  # Use custom collate function
# # # # # # # )


# # # # # # # # JAX function
# # # # # # # @jax.jit
# # # # # # # def loss_fn(params, batch):
# # # # # # #     logits = batch["features"] @ params
# # # # # # #     return jnp.mean((logits - batch["labels"]) ** 2)


# # # # # # # # Dummy params
# # # # # # # params = jnp.zeros((2, 1))

# # # # # # # # Training loop
# # # # # # # for batch in dataloader:
# # # # # # #     # Convert PyTorch tensors to JAX arrays
# # # # # # #     batch = jax.tree.map(lambda x: jax.device_put(x.numpy()), batch)
# # # # # # #     loss = loss_fn(params, batch)
# # # # # # #     print("Batch features:", batch["features"])
# # # # # # #     print("Batch labels:", batch["labels"])
# # # # # # #     print("Batch metadata:", batch["metadata"])
# # # # # # #     print("Loss:", loss)


# # # # # # # import torch
# # # # # # # from torch.utils.data import Dataset, DataLoader
# # # # # # # import jax
# # # # # # # import jax.numpy as jnp
# # # # # # # import jax.tree_util as jtu

# # # # # # # # Example PyTree dataset
# # # # # # # dataset = {
# # # # # # #     "features": jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=jnp.float32),
# # # # # # #     "labels": jnp.array([0, 1, 0, 1], dtype=jnp.float32),
# # # # # # #     "metadata": {"id": jnp.array([10, 20, 30, 40], dtype=jnp.int32)},
# # # # # # # }


# # # # # # # class PyTreeDataset(Dataset):
# # # # # # #     def __init__(self, pytree_data):
# # # # # # #         self.data = pytree_data
# # # # # # #         # Generalize n_samples using tree_leaves
# # # # # # #         leaves = jtu.tree_leaves(pytree_data)
# # # # # # #         if not leaves:
# # # # # # #             raise ValueError("PyTree has no leaves!")
# # # # # # #         self.n_samples = len(leaves[0])  # Assume consistent length across leaves

# # # # # # #     def __len__(self):
# # # # # # #         return self.n_samples

# # # # # # #     def __getitem__(self, idx):
# # # # # # #         # Return a PyTree of JAX arrays
# # # # # # #         sample = jax.tree_map(lambda x: x[idx], self.data)
# # # # # # #         return sample


# # # # # # # # Custom collate function to batch PyTrees with JAX arrays
# # # # # # # def jax_collate_fn(batch):
# # # # # # #     # Batch is a list of PyTree samples; stack them using JAX
# # # # # # #     batched = jax.tree_map(lambda *xs: jnp.stack(xs), *batch)
# # # # # # #     return batched


# # # # # # # # Create DataLoader
# # # # # # # pytorch_dataset = PyTreeDataset(dataset)
# # # # # # # dataloader = DataLoader(
# # # # # # #     pytorch_dataset,
# # # # # # #     batch_size=2,
# # # # # # #     shuffle=True,
# # # # # # #     collate_fn=jax_collate_fn,  # Use JAX-based collation
# # # # # # # )


# # # # # # # # JAX function
# # # # # # # @jax.jit
# # # # # # # def loss_fn(params, batch):
# # # # # # #     logits = batch["features"] @ params
# # # # # # #     return jnp.mean((logits - batch["labels"]) ** 2)


# # # # # # # # Dummy params
# # # # # # # params = jnp.zeros((2, 1))

# # # # # # # # Training loop
# # # # # # # for batch in dataloader:
# # # # # # #     # No conversion needed—batch is already a PyTree of JAX arrays
# # # # # # #     loss = loss_fn(params, batch)
# # # # # # #     print("Batch features:", batch["features"])
# # # # # # #     print("Batch labels:", batch["labels"])
# # # # # # #     print("Batch metadata:", batch["metadata"])
# # # # # # #     print("Loss:", loss)


# # # # # # import torch
# # # # # # from torch.utils.data import Dataset, DataLoader
# # # # # # import jax
# # # # # # import jax.numpy as jnp
# # # # # # import jax.tree_util as jtu
# # # # # # import numpy as np
# # # # # # import os

# # # # # # # Simulate a large dataset (e.g., 1 GB of features)
# # # # # # n_samples = 100_000  # 100k samples
# # # # # # feature_dim = 2_500  # 2.5k floats per sample
# # # # # # # 100k * 2.5k * 4 bytes (float32) = ~1 GB

# # # # # # # Create and save large memory-mapped arrays
# # # # # # if not os.path.exists("large_features.npy"):
# # # # # #     large_features = np.random.randn(n_samples, feature_dim).astype(np.float32)
# # # # # #     np.save("large_features.npy", large_features)
# # # # # # if not os.path.exists("large_labels.npy"):
# # # # # #     large_labels = np.random.randint(0, 2, n_samples).astype(np.float32)
# # # # # #     np.save("large_labels.npy", large_labels)

# # # # # # # PyTree with memory-mapped arrays
# # # # # # dataset = {
# # # # # #     "features": jnp.array(np.memmap("large_features.npy", dtype=np.float32, mode="r", shape=(n_samples, feature_dim))),
# # # # # #     "labels": jnp.array(np.memmap("large_labels.npy", dtype=np.float32, mode="r", shape=(n_samples,))),
# # # # # # }


# # # # # # class PyTreeDataset(Dataset):
# # # # # #     def __init__(self, pytree_data):
# # # # # #         self.data = pytree_data
# # # # # #         leaves = jtu.tree_leaves(pytree_data)
# # # # # #         if not leaves:
# # # # # #             raise ValueError("PyTree has no leaves!")
# # # # # #         self.n_samples = len(leaves[0])

# # # # # #     def __len__(self):
# # # # # #         return self.n_samples

# # # # # #     def __getitem__(self, idx):
# # # # # #         # Log when data is accessed
# # # # # #         print(f"Loading index {idx}")
# # # # # #         sample = jax.tree_map(lambda x: x[idx], self.data)
# # # # # #         return sample


# # # # # # # Custom collate function for JAX arrays
# # # # # # def jax_collate_fn(batch):
# # # # # #     return jax.tree_map(lambda *xs: jnp.stack(xs), *batch)


# # # # # # # Create DataLoader
# # # # # # pytorch_dataset = PyTreeDataset(dataset)
# # # # # # dataloader = DataLoader(pytorch_dataset, batch_size=2, shuffle=True, collate_fn=jax_collate_fn)


# # # # # # # JAX function
# # # # # # @jax.jit
# # # # # # def loss_fn(params, batch):
# # # # # #     logits = batch["features"] @ params
# # # # # #     return jnp.mean((logits - batch["labels"]) ** 2)


# # # # # # # Dummy params (adjust size for large features)
# # # # # # params = jnp.zeros((feature_dim, 1))

# # # # # # # Iterate over just a few batches to prove lazy loading
# # # # # # for i, batch in enumerate(dataloader):
# # # # # #     if i >= 2:  # Limit to 2 batches for brevity
# # # # # #         break
# # # # # #     loss = loss_fn(params, batch)
# # # # # #     print(f"Batch {i} features shape: {batch['features'].shape}")
# # # # # #     print(f"Batch {i} labels: {batch['labels']}")
# # # # # #     print(f"Loss: {loss}")


# # # # # import wandb
# # # # # import random

# # # # # # Initialize wandb run
# # # # # wandb.init(project="multiline_test")

# # # # # # Define columns
# # # # # columns = ["T (degC)", "p (mbar)", "rho (g/m**3)"]

# # # # # # First set of data (0 to 199 steps)
# # # # # num_steps = 200
# # # # # xs1 = [i for i in range(num_steps)]
# # # # # ys1 = [
# # # # #     [random.uniform(15, 25) for _ in range(num_steps)],  # T (degC)
# # # # #     [random.uniform(900, 1100) for _ in range(num_steps)],  # p (mbar)
# # # # #     [random.uniform(1.1, 1.3) for _ in range(num_steps)],  # rho (g/m**3)
# # # # # ]

# # # # # # Log first plot
# # # # # wandb.log({"weather_sample": wandb.plot.line_series(xs=xs1, ys=ys1, keys=columns, title="Weather Metrics")})

# # # # # # Second set of data (200 to 299 steps) - no combining with first set
# # # # # extra_steps = 100
# # # # # xs2 = [i for i in range(num_steps, num_steps + extra_steps)]
# # # # # ys2 = [
# # # # #     [random.uniform(15, 25) for _ in range(extra_steps)],
# # # # #     [random.uniform(900, 1100) for _ in range(extra_steps)],
# # # # #     [random.uniform(1.1, 1.3) for _ in range(extra_steps)],
# # # # # ]

# # # # # # Log second plot
# # # # # wandb.log({"weather_sample": wandb.plot.line_series(xs=xs2, ys=ys2, keys=columns, title="Weather Metrics")})

# # # # # wandb.finish()


# # # # import wandb
# # # # import os

# # # # # First run
# # # # run1 = wandb.init(project="test-project1", name="run-1")

# # # # # Create file1.txt for the first run
# # # # with open("file1.txt", "w") as f:
# # # #     f.write("Hello from Run 1!")

# # # # artifact1 = wandb.Artifact(name="my-artifact", type="dataset")
# # # # artifact1.add_file("file1.txt")
# # # # run1.log_artifact(artifact1)
# # # # run1.finish()

# # # # # Second run
# # # # run2 = wandb.init(project="test-project1", name="run-2")

# # # # # Create file2.txt for the second run
# # # # with open("file2.txt", "w") as f:
# # # #     f.write("Hello from Run 2!")

# # # # artifact2 = wandb.Artifact(name="my-artifact", type="dataset")
# # # # artifact2.add_file("file2.txt")
# # # # run2.log_artifact(artifact2)
# # # # run2.finish()

# # # # # Clean up (optional)
# # # # os.remove("file1.txt")
# # # # os.remove("file2.txt")


# # # import wandb
# # # import os

# # # # Step 1: Create and upload the artifact in online mode
# # # with wandb.init(project="offline-experiment", mode="online") as run:
# # #     with open("final_model_weights.pt", "w") as f:
# # #         f.write("Dummy model weights")

# # #     artifact = wandb.Artifact("autoprior_celeba_64", type="model")
# # #     artifact.add_file("final_model_weights.pt")
# # #     artifact.metadata = {"input_size": 64, "channels": 3}
# # #     run.log_artifact(artifact)
# # #     run.finish()
# # #     api = wandb.Api()
# # #     uploaded_artifact = api.artifact("wlp9800-new-york-university/offline-experiment/autoprior_celeba_64:latest")
# # #     uploaded_artifact.aliases.append("v19")
# # #     uploaded_artifact.save()
# # #     print("Artifact 'wlp9800-new-york-university/offline-experiment/autoprior_celeba_64:v19' created")

# # # # Step 2: Try to download in offline mode without a with statement
# # # run = wandb.init(project="offline-experiment", mode="offline")
# # # print("Run initialized in offline mode")
# # # api = wandb.Api()
# # # model_artifact = api.artifact("wlp9800-new-york-university/offline-experiment/autoprior_celeba_64:latest")
# # # model_dir = model_artifact.download()
# # # model_path = f"{model_dir}/final_model_weights.pt"
# # # config = model_artifact.metadata
# # # print(f"Artifact downloaded to: {model_dir}")
# # # print(f"Model path: {model_path}")
# # # print(f"Config: {config}")
# # # run.finish()

# # # # Clean up
# # # os.remove("final_model_weights.pt")

# # import jax
# # import jax.numpy as jnp
# # from jax import tree


# # # https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
# # def tree_stack(trees):
# #     return jax.tree.map(lambda *v: jnp.stack(v), *trees)


# # def tree_unstack(tree):
# #     leaves, treedef = jax.tree.flatten(tree)
# #     return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]


# # # Test code
# # def test_tree_stack_unstack():
# #     # Create sample PyTrees with different structures
# #     tree1 = {"a": jnp.array([1, 2, 3]), "b": {"x": jnp.array([4, 5])}, "c": 7}

# #     tree2 = {"a": jnp.array([10, 20, 30]), "b": {"x": jnp.array([40, 50])}, "c": 8}

# #     tree3 = {"a": jnp.array([100, 200, 300]), "b": {"x": jnp.array([400, 500])}, "c": 9}

# #     # List of trees to stack
# #     trees = [tree1, tree2, tree3]

# #     # Stack the trees
# #     stacked = tree_stack(trees)

# #     # Print original trees
# #     print("Original trees:")
# #     for i, t in enumerate(trees):
# #         print(f"Tree {i}:", t)

# #     # Print stacked result
# #     print("\nStacked result:")
# #     print(stacked)

# #     # Unstack back to individual trees
# #     unstacked = tree_unstack(stacked)

# #     # Print unstacked results
# #     print("\nUnstacked trees:")
# #     for i, t in enumerate(unstacked):
# #         print(f"Unstacked tree {i}:", t)

# #     # Verify the unstacked trees match the originals
# #     for orig, unst in zip(trees, unstacked):

# #         def check_equal(t1, t2):
# #             return jax.tree_util.tree_all(jax.tree_map(lambda x, y: jnp.all(x == y), t1, t2))

# #         assert check_equal(orig, unst), "Unstacked tree doesn't match original!"

# #     print("\nVerification: All unstacked trees match their originals!")


# # # Run the test
# # if __name__ == "__main__":
# #     test_tree_stack_unstack()


# import wandb
# import numpy as np
# import time

# # Start W&B
# wandb.init(project="test-heavy-logging")

# # Fake a hefty payload: a 1000x1000 random array (like an image or tensor)
# big_data = np.random.rand(1000, 1000)

# for i in range(50):  # Fewer iterations, but heavier logs
#     start = time.time()

#     # Log something intensive: a big array and some metrics
#     wandb.log(
#         {
#             "step": i,
#             "big_array": big_data,  # 1M floats, ~8MB raw
#             "fake_image": wandb.Image(np.random.rand(256, 256)),  # Mimic image logging
#             "loss": i * 0.1,
#         }
#     )

#     end = time.time()
#     print(f"Step {i}, log time: {end - start:.6f} secs")

#     # Simulate training work
#     # time.sleep(0.1)  # Pretend we're doing a batch


import jax
import jax.flatten_util
import jax.numpy as jnp
import optax


def soft_clip_norm(threshold: float):
    """Applies a soft gradient norm clipping transformation."""

    def update_fn(updates, state, _):
        # Flatten the gradient updates
        grads_flat, unravel_fn = jax.flatten_util.ravel_pytree(updates)

        # Compute L2 norm
        grad_norm = jnp.linalg.norm(grads_flat, ord=2)

        # Apply soft clipping
        clipped_norm = grad_norm - jax.nn.softplus(grad_norm - threshold)

        # Compute scale factor
        scale = clipped_norm / (grad_norm + 1e-6)

        # Scale the updates
        updates_scaled = jax.tree_util.tree_map(lambda g: g * scale, updates)

        return updates_scaled, state

    return optax.GradientTransformation(lambda _: (), update_fn)


# Define a simple model (linear regression)
def loss_fn(params, x, y):
    pred = params["w"] * x + params["b"]
    return jnp.mean((pred - y) ** 2)


# Dummy dataset
x_data = jnp.array([1.0, 2.0, 3.0])
y_data = jnp.array([2.0, 4.0, 6.0])  # y = 2x (perfect linear function)

# Initialize parameters
params = {"w": jnp.array(3.0), "b": jnp.array(0.0)}

# Define optimizer with soft clipping
threshold = 5.0  # Soft clip threshold
optimizer = optax.chain(
    soft_clip_norm(threshold),  # Apply soft gradient clipping
    optax.sgd(learning_rate=0.1),  # Use SGD optimizer
)
opt_state = optimizer.init(params)


# Training step
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss, grads, updates


# Run one training step
params, opt_state, loss, grads, updates = train_step(params, opt_state, x_data, y_data)

# Compute gradient norms
grad_norm_before = jnp.linalg.norm(jax.flatten_util.ravel_pytree(grads)[0])
grad_norm_after = jnp.linalg.norm(jax.flatten_util.ravel_pytree(updates)[0])

print(f"Loss: {loss:.4f}")
print(f"Gradient norm before: {grad_norm_before:.4f}")
print(f"Gradient norm after:  {grad_norm_after:.4f}")
