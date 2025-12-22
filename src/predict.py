import pandas as pd
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join("..", "src")))

from shoe_dataset import ShoeDataset
from shoe_classifier import ShoeClassifier
from shared import get_num_classes, get_transforms
from gradcam import generate_gradcam_heatmap

def extract_features_from_dataset(model, dataset, device):
    features = []
    labels = []
    model.eval()
    print(f"Extracting features from {len(dataset)} images...")
    with torch.no_grad():
        for i in range(len(dataset)):
            img, label = dataset[i]
            if img is not None:
                img = img.unsqueeze(0).to(device)
                feature = model.extract_features(img).cpu().numpy()
                features.append(feature)
                labels.append(label.item())
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(dataset)}")
    
    if features:
        features = np.concatenate(features, axis=0)
    return features, np.array(labels)

def find_similar_shoes(input_features, features, labels, top_k=5):
    similarities = cosine_similarity(input_features, features)
    similar_indices = np.argsort(-similarities[0])[:top_k]
    similar_labels = labels[similar_indices]
    return similar_labels

def predict_and_recommend(model, image_path, label_encoder, features, labels, device, top_k=5):
    transform = get_transforms()
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return None, None

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        _, predicted = torch.max(output, 1)
    
    predicted_idx = predicted.item()
    predicted_model = label_encoder.inverse_transform([predicted_idx])[0]

    print("\n--- Top 5 Predictions ---")
    for i in range(5):
        idx = top5_catid[0][i].item()
        prob = top5_prob[0][i].item()
        class_name = label_encoder.inverse_transform([idx])[0]
        print(f"{i+1}. {class_name}: {prob*100:.2f}%")
    print("------------------------------------\n")
    
    # Recommendation
    input_features = model.extract_features(image_tensor).cpu().numpy()
    similar_labels_idx = find_similar_shoes(input_features, features, labels, top_k)
    similar_models = label_encoder.inverse_transform(similar_labels_idx)
    
    # Grad-CAM
    heatmap = generate_gradcam_heatmap(model, image_tensor, predicted_idx)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_model}")
    plt.axis('off')
    
    if heatmap is not None:
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        heatmap = cv2.resize(heatmap, (image.width, image.height))
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.title("Grad-CAM Heatmap")
        plt.axis('off')
    
    plt.tight_layout()
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "plots")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "prediction_heatmap.png")
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    
    return predicted_model, similar_models

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = get_num_classes()
    model = ShoeClassifier(num_classes=num_classes)
    model.to(device)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "outputs", "models", "RunningShoesClassifier.pth")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded.")
    else:
        print(f"Model file not found at {model_path}. Please train the model first.")
        sys.exit(1)

    model.eval()

    print("Loading dataset for feature extraction...")
    data_csv_path = os.path.join(current_dir, "..", "data", "Running Shoes Data.csv")
    if not os.path.exists(data_csv_path):
        print(f"Data file not found at {data_csv_path}")
        sys.exit(1)
        
    df = pd.read_csv(data_csv_path)
    
    def fix_path(path):
        return os.path.normpath(os.path.join(current_dir, path))

    df.iloc[:, 0] = df.iloc[:, 0].apply(fix_path)
    transform = get_transforms()
    full_dataset = ShoeDataset(df, transform=transform)
    label_encoder = full_dataset.label_encoder

    train_df, _ = train_test_split(df, test_size=0.2, random_state=42)
    
    train_df = train_df.head(100) 
    

    
    train_dataset = ShoeDataset(train_df, transform=transform)

    train_dataset.label_encoder = label_encoder
    train_dataset.dataframe["model_encoded"] = label_encoder.transform(train_dataset.dataframe["model"])

    print("Extracting features from dataset...")
    features, labels = extract_features_from_dataset(model, train_dataset, device)
    print(f"Extracted features shape: {features.shape}")

    parser = argparse.ArgumentParser(description="Predict shoe model and recommend similar shoes.")
    parser.add_argument("--image", type=str, help="Path to the input image for prediction.")
    args = parser.parse_args()

    test_img_path = args.image

    if test_img_path is None:
        test_img_path = os.path.join(current_dir, "..", "data", "Adidas Ultraboost", "image.jpeg")
        if not os.path.exists(test_img_path):
            if len(df) > 0:
                test_img_path = df.iloc[0, 0]
    
    if not os.path.exists(test_img_path):
        print(f"No test image found at {test_img_path}")
    else:
        print(f"Predicting for {test_img_path}...")
        pred, recs = predict_and_recommend(model, test_img_path, label_encoder, features, labels, device)
        
        if pred:
            print(f"Predicted Model: {pred}")
            print("Similar Models Recommendations:")
            for i, rec in enumerate(recs):
                print(f"{i+1}: {rec}")