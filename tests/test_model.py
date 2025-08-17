# Unit tests for the model defined in model.py.

import pytest
from src.deep_learning.model import NeuralNetwork

def test_model_initialization():
    model = NeuralNetwork()
    assert model is not None

def test_model_forward_pass():
    model = NeuralNetwork()
    # Create a dummy input tensor
    input_tensor = torch.randn(1, 3, 224, 224)  # Example input shape for an image
    output = model(input_tensor)
    assert output is not None  # Ensure output is not None
    assert output.shape == (1, 3, 224, 224)  # Adjust based on your model's output shape

def test_model_training():
    model = NeuralNetwork()
    # Dummy data and labels
    dummy_data = torch.randn(10, 3, 224, 224)  # Batch of 10 images
    dummy_labels = torch.randint(0, 2, (10,))  # Binary labels for classification

    # Define a simple loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Forward pass
    outputs = model(dummy_data)
    loss = criterion(outputs, dummy_labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss.item() >= 0  # Loss should be non-negative after training step