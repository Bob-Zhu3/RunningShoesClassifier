import pandas as pd

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import sys
import os

sys.path.append(os.path.abspath(os.path.join("..", "src")))

from shoe_dataset import ShoeDataset
from shoe_classifier import ShoeClassifier

from shared import get_num_classes, get_transforms

model = ShoeClassifier(num_classes=get_num_classes())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.load_state_dict(torch.load("../outputs/models/RunningShoesClassifier.pth"))
model.eval()

df = pd.read_csv("../data/Running Shoes Data.csv")
_, test_df = train_test_split(df, test_size=0.2, random_state=42)

transform = get_transforms()
test_dataset = ShoeDataset(test_df, transform=transform)
test_data = [(img, label) for img, label in test_dataset if img is not None and label is not None]
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")