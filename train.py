from src.dataset import get_dataloader # type: ignore
from src.model import Classifier # type: ignore
from src.symbolic_module import apply_rules # type: ignore
from src.trainer import train_model # type: ignore
import torch

def main():
    # Load data
    train_loader, test_loader = get_dataloader(batch_size=32)

    # Init model
    model = Classifier(input_dim=28*28, hidden_dim=128, output_dim=10)

    # Train
    train_model(model, train_loader, test_loader, epochs=3)

if __name__ == "__main__":
    main()
