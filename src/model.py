import torch
import torch.nn as nn
import torchvision

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 1600)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

def get_model(config):
    if config["model"] == "SimpleCNN":
        return SimpleCNN()
    elif config["model"] == "ResNet18":
        model = torchvision.models.resnet18(weights=None)
        model.fc = nn.Linear(512, 10)
        return model
    else:
        raise ValueError("Model not supported")