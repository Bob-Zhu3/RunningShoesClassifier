import os
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class ShoeDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.dataframe["model_encoded"] = self.label_encoder.fit_transform(self.dataframe["model"])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        if not os.path.exists(img_path):
            print(f"Could not identify image file {img_path}. Skipping.")
            return None, None
        try:
            image = Image.open(img_path).convert("RGB")
        except UnidentifiedImageError:
            print(f"Could not identify image file {img_path}. Skipping.")
            return None, None
        label = self.dataframe.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        print(f"Loaded image {img_path} with label {label}")
        return image, torch.tensor(label, dtype=torch.long)