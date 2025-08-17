import torch
from torchvision import datasets, transforms

def get_dataloader(batch_size=32):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.view(-1))])  # Flatten
    train_data = datasets.MNIST(root="data/raw", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="data/raw", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
