import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import sys
import os

sys.path.append(os.path.abspath(os.path.join("..", "src")))

from shoe_dataset import ShoeDataset
from shoe_classifier import ShoeClassifier

from shared import get_num_classes, get_transforms

df = pd.read_csv("../data/Running Shoes Data.csv")
train_df, _ = train_test_split(df, test_size=0.2, random_state=42)

transform = get_transforms()
train_dataset = ShoeDataset(train_df, transform=transform)
train_data = [(img, label) for img, label in train_dataset if img is not None and label is not None]
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)

num_classes = get_num_classes()
model = ShoeClassifier(num_classes=num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Start training...")
epochs = 20
losses = []

print("Number of batches:", len(train_loader))

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_dataset)
    losses.append(epoch_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

plt.plot(range(1, epochs + 1), losses, marker="o")
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
plt.savefig("../outputs/plots/training_loss_curve.png")

torch.save(model.state_dict(), "../outputs/models/RunningShoesClassifier.pth")
print("Saved model to RunningShoesClassifier.pth")