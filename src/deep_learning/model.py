import torch
import torch.nn as nn

# This module defines the neural network architecture for the project.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Define layers here
        self.fc1 = nn.Linear(10, 50)  # Example layer
        self.fc2 = nn.Linear(50, 1)    # Example layer

    def forward(self, x):
        # Define forward pass
        x = torch.relu(self.fc1(x))    # Example activation
        x = self.fc2(x)
        return x

    def train_model(self, data_loader, criterion, optimizer, num_epochs):
        # Training loop for the model
        for epoch in range(num_epochs):
            for inputs, labels in data_loader:
                optimizer.zero_grad()        # Zero the gradients
                outputs = self.forward(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                loss.backward()              # Backward pass
                optimizer.step()             # Update weights
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')  # Print loss per epoch