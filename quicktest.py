import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)

# Define the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # RNN forward pass
        out, _ = self.rnn(x, h0)
        # Decode the last time step's output
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = 10     # Number of input features
hidden_size = 20    # Number of hidden units
num_layers = 2      # Number of RNN layers
output_size = 2     # Number of output classes
sequence_length = 15  # Length of input sequences
batch_size = 8      # Number of sequences per batch
num_epochs = 30     # Number of training epochs
learning_rate = 0.001

# Create the model, loss function, and optimizer
model = SimpleRNN(input_size, hidden_size, num_layers, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Generate random data
x_train = torch.randn(100, sequence_length, input_size)  # 100 training samples
y_train = torch.randint(0, output_size, (100,))          # Random labels
x_test = torch.randn(20, sequence_length, input_size)    # 20 test samples
y_test = torch.randint(0, output_size, (20,))            # Random labels

# Train the model
for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        # Forward pass
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(x_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).float().mean()
    print(f"Test Accuracy: {accuracy:.4f}")
