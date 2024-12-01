from dataclasses import dataclass
from typing import Callable, Generic, TypeVar
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


@dataclass
class RnnConfig:
    n_in: int
    n_h: int
    n_out: int
    num_layers: int

T = TypeVar('T')
X = TypeVar('X')

@dataclass
class RNNInterface(Generic[X, T]):
    baseCase: Callable[[X], T]
    forward: Callable[[X, T], torch.Tensor]

# Fully connected neural network with one hidden layer
class RNN(nn.Module):
    def __init__(self, config: RnnConfig):
        super(RNN, self).__init__()
        self.fc = nn.Linear(config.n_h, config.n_out)

        self.rnn = nn.RNN(config.n_in, config.n_h, config.num_layers, batch_first=True)
        self.interface = RNNInterface(
            baseCase=lambda x: torch.zeros(config.num_layers, x.size(0), config.n_h),
            forward=lambda x, s0: self.rnn(x, s0)[0][:, -1, :]
        )
        # self.gru = nn.GRU(config.n_in, config.n_h, config.num_layers, batch_first=True)
        # self.lstm = nn.LSTM(config.n_in, config.n_h, config.num_layers, batch_first=True)
        # self.interface = RNNInterface(
        #     baseCase=lambda x: (torch.zeros(config.num_layers, x.size(0), config.n_h), torch.zeros(self.num_layers, x.size(0), self.hidden_size)),
        #     forward=lambda x, s0: self.lstm(x, s0)[0]
        # )
        
    def forward(self, x):
        s0 = self.interface.baseCase(x)
        out = self.interface.forward(x, s0)
        out = self.fc(out)
        return out





# Hyper-parameters 
# input_size = 784 # 28x28
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


rnnConfig = RnnConfig(
    n_in=28,
    n_h=128,
    n_out=10,
    num_layers=2
)

model = RNN(rnnConfig)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # origin shape: [N, 1, 28, 28]
        # resized: [N, 28, 28]
        images = images.squeeze(1)
        
        # Forward pass
        outputs = model(images)

        # print(images.shape)
        # print(labels.shape)
        # print(outputs.shape)
        # quit()

        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.squeeze(1)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')