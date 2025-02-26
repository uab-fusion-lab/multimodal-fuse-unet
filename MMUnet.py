import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from data_prepare import RegDBDualModalDataset
from model import DualModalUNet

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images if needed
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])


index_files = [f'data/RegDB/idx/train_thermal_{i}.txt' for i in range(1, 11)]

train_dataset = RegDBDualModalDataset(data_root='data/RegDB',
                                      index_files=index_files,
                                      transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = DualModalUNet()
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # Binary Cross-Entropy Loss for binary segmentation
optimizer = optim.Adam(model.parameters(), lr=0.001)

# for thermal_images, visible_images, labels in train_loader:
#     print(labels)

test_index_files = [f'data/RegDB/idx/test_thermal_{i}.txt' for i in range(1, 11)]
test_dataset = RegDBDualModalDataset(data_root='data/RegDB',
                                      index_files=test_index_files,
                                      transform=transform)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# for thermal_images, visible_images, labels in test_loader:
#     print(labels)

def test():
    model.eval()
    model.to(device)


    all_predictions = []
    all_labels = []


    with torch.no_grad():
        for thermal_images, visible_images, labels in test_loader:
            thermal_images = thermal_images.to(device)
            visible_images = visible_images.to(device)
            labels = labels.to(device)


            outputs = model(thermal_images, visible_images)
            _, predicted = torch.max(outputs, 1)


            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Accuracy: {accuracy:.4f}")


def add_gaussian_noise(tensor, mean=0., std=1.):
    """
    Adds Gaussian noise to a tensor.

    Parameters:
        tensor (torch.Tensor): The input tensor (image).
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: The noisy tensor (image).
    """
    noise = torch.randn(tensor.size()) * std + mean
    noisy_tensor = tensor + noise
    return noisy_tensor#.clamp(0, 1)

def train_model(model, train_loader, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for thermal_images, visible_images, labels in train_loader:
            thermal_images = add_gaussian_noise(thermal_images, std=1)
            thermal_images = thermal_images.to(device)
            visible_images = visible_images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(thermal_images, visible_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
        test()



num_epochs = 100
train_model(model, train_loader, optimizer, num_epochs)


