from dataclasses import dataclass
import itertools 
import torch

test_3d_tesnor = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 1, 1]], [[10, 11, 12], [13, 14, 15], [16, 17, 18], [1, 1, 1]]])
test = torch.chunk(test_3d_tesnor, 2, dim=1)
print(test)