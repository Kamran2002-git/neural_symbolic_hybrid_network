import torch
from src.dataset import get_dataloader
from src.model import Classifier
from src.evaluate import evaluate_model

def main():
    _, test_loader = get_dataloader(batch_size=32)
    model = Classifier(28*28, 128, 10)
    model.load_state_dict(torch.load("model.pth"))
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()
