import pandas as pd

import torch
from PIL import Image

import sys
import os

sys.path.append(os.path.abspath(os.path.join("..", "src")))

from shoe_dataset import ShoeDataset
from shoe_classifier import ShoeClassifier

from shared import get_num_classes, get_transforms

def predict(model, image_path, label_encoder):
    transform = get_transforms()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return label_encoder.inverse_transform(predicted.cpu().numpy())[0]

model = ShoeClassifier(num_classes=get_num_classes())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.load_state_dict(torch.load("../outputs/models/RunningShoesClassifier.pth"))
model.eval()

df = pd.read_csv("../data/Running Shoes Data.csv")
train_df = df.sample(frac=0.8, random_state=42)
label_encoder = ShoeDataset(train_df).label_encoder

img = "path/to/image"
predicted_model = predict(model, img, label_encoder)
print(f"Predicted Model: {predicted_model}")