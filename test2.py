import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler


# Custom Dataset
class MyDataset(Dataset):
    def __init__(self, size=10):
        self.data = list(range(size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Create a dataset
dataset = MyDataset()

# Create the RandomSampler
sampler = RandomSampler(dataset)

# Create DataLoader with RandomSampler


# Loop over multiple epochs
for epoch in range(3):
    print(f"Epoch {epoch + 1}")
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=5)

    for batch_idx, data in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}: {data.tolist()}")

    print("-" * 40)
