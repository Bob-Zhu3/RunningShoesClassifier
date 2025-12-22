import pandas as pd
from sklearn.model_selection import train_test_split

import sys
import os

sys.path.append(os.path.abspath(os.path.join("..", "src")))

from shoe_dataset import ShoeDataset
from image_transforms import get_transforms

def get_num_classes(data_path=None):
    if data_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "..", "data", "Running Shoes Data.csv")

    df = pd.read_csv(data_path)
    label_encoder = ShoeDataset(df, transform=None).label_encoder
    return len(label_encoder.classes_)