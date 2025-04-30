from torch.utils.data import Dataset
import jax
import jax.numpy as jnp


class PyTreeDataset(Dataset):
    def __init__(self, pytree_data):
        self.data = pytree_data
        leaves = jax.tree.leaves(pytree_data)
        if not leaves:
            raise ValueError("PyTree has no leaves!")
        self.n_samples = len(leaves[0])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return jax.tree.map(lambda x: x[idx], self.data)


def jax_collate_fn(batch):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *batch)


# Transforms
def flatten_and_cast(pic, x_pixels):
    """Convert image to flat (y, x_pixels) JAX array."""
    arr = jnp.array(pic, dtype=jnp.float32) / 255.0  # normalize
    flat = arr.ravel()  # flatten in scanline order (row-major)
    total_pixels = flat.shape[0]

    if total_pixels % x_pixels != 0:
        raise ValueError(f"Cannot reshape array of size {total_pixels} into shape (-1, {x_pixels}).")

    return flat.reshape(-1, x_pixels)


def target_transform(label, sequence_length):
    """Convert scalar label to (784, 2) JAX array: [class_label, sequence_number]."""
    labels = jnp.zeros((sequence_length, 2), dtype=jnp.int32)
    labels = labels.at[:, 0].set(label)  # Repeat class label
    labels = labels.at[:, 1].set(jnp.arange(sequence_length))  # Sequence numbers
    return labels
