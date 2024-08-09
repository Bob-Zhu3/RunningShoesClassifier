import pandas as pd
from sklearn.model_selection import train_test_split

import sys
import os

sys.path.append(os.path.abspath(os.path.join("..", "src")))

from shoe_dataset import ShoeDataset
from image_transforms import get_transforms

def get_num_classes(data_path="../data/Running Shoes Data.csv"):
    df = pd.read_csv(data_path)
    train_df, _ = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = ShoeDataset(train_df, transform=get_transforms())
    return len(train_dataset.label_encoder.classes_)