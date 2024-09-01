import numpy as np

def extract_features_from_dataset(model, dataset):
    features = []
    labels = []
    for img, label in dataset:
        if img is not None:
            img = img.unsqueeze(0).to(device)
            feature = model.extract_features(img).cpu().numpy()
            features.append(feature)
            labels.append(label.item())
    return np.array(features), np.array(labels)

features, labels = extract_features_from_dataset(model, train_dataset)

from sklearn.metrics.pairwise import cosine_similarity

def find_similar_shoes(input_features, features, labels, top_k=5):
    similarities = cosine_similarity(input_features, features)
    similar_indices = np.argsort(-similarities[0])[:top_k]
    similar_labels = labels[similar_indices]
    return similar_labels
