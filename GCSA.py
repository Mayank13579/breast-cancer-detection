import ssl
import urllib3
import numpy as np
import pandas as pd
import cv2
import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from tensorflow.keras.applications import ResNet50, InceptionV3, VGG16
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification

# Disable SSL warnings
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ========== Step 1: Data Preprocessing ==========
DATASET_PATH = './dataset/'
image_size = (224, 224)

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, image_size)
    image = image / 255.0
    return image

X, y = [], []
for label in ['IDC_negative', 'IDC_positive']:
    folder_path = os.path.join(DATASET_PATH, label)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        image = load_and_preprocess_image(img_path)
        if image is not None:
            X.append(image)
            y.append(0 if label == 'IDC_negative' else 1)

X = np.array(X)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== Step 2: Feature Extraction (CNN Models) ==========
def extract_features_cnn(model, data):
    features = model.predict(data)
    return features.reshape((features.shape[0], -1))

cnn_models = [
    ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
]

cnn_features_train = [extract_features_cnn(model, X_train) for model in cnn_models]
cnn_features_train = np.concatenate(cnn_features_train, axis=1)
cnn_features_test = [extract_features_cnn(model, X_test) for model in cnn_models]
cnn_features_test = np.concatenate(cnn_features_test, axis=1)

# ========== Step 3: Feature Extraction (Transformer Models) ==========
processor_vit = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", do_rescale=False)
model_vit = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k").eval()

processor_swin = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224", do_rescale=False)
model_swin = AutoModelForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224").eval()

def extract_features_transformer(processor, model, images, batch_size=16):
    features = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        inputs = processor(batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            try:
                outputs = model(**inputs)
                if hasattr(outputs, 'last_hidden_state'):
                    batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                else:
                    batch_features = outputs.logits.cpu().numpy()
                features.append(batch_features)
            except RuntimeError as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                continue

    return np.concatenate(features, axis=0)

transformer_features_train_vit = extract_features_transformer(processor_vit, model_vit, X_train)
transformer_features_test_vit = extract_features_transformer(processor_vit, model_vit, X_test)
transformer_features_train_swin = extract_features_transformer(processor_swin, model_swin, X_train)
transformer_features_test_swin = extract_features_transformer(processor_swin, model_swin, X_test)

transformer_features_train = np.concatenate([transformer_features_train_vit, transformer_features_train_swin], axis=1)
transformer_features_test = np.concatenate([transformer_features_test_vit, transformer_features_test_swin], axis=1)

# Combine features
# features_train = np.concatenate((cnn_features_train, transformer_features_train), axis=1)    #OLD feature extraction code
# features_test = np.concatenate((cnn_features_test, transformer_features_test), axis=1)


# Combine features from CNN and Transformer
features_train = np.concatenate((cnn_features_train, transformer_features_train), axis=1)
features_test = np.concatenate((cnn_features_test, transformer_features_test), axis=1)

# ========================= Apply PCA Here =========================
from sklearn.decomposition import PCA
import torch

# Set the number of PCA components (tunable)
n_pca_components = 256

# Apply PCA to reduce feature dimensions
pca = PCA(n_components=n_pca_components)

# Transform training and testing features
features_train = pca.fit_transform(features_train)
features_test = pca.transform(features_test)

# Convert back to tensors for compatibility with PyTorch
features_train = torch.tensor(features_train, dtype=torch.float32)
features_test = torch.tensor(features_test, dtype=torch.float32)
# ===============================================================

# ========== Step 4: Gated Channel Self-Attention (GCSA) ==========
class GCSA(nn.Module):
    def __init__(self, feature_dim):
        super(GCSA, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.conv1 = nn.Conv1d(feature_dim, feature_dim, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(feature_dim, feature_dim, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(feature_dim, feature_dim, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Ensure it has correct dimensions
        q = self.conv1(x)
        k = self.conv2(x)
        v = self.conv3(x)
        attention = self.softmax(torch.bmm(q.transpose(1, 2), k))
        out = torch.bmm(v, attention)
        out = self.gamma * out + x
        return out.squeeze().detach().cpu().numpy() 



gcsa = GCSA(feature_dim=features_train.shape[1])

# Ensure features_train and features_test retain the same shape
features_train = np.array([gcsa(torch.tensor(f, dtype=torch.float32)) for f in features_train])
features_test = np.array([gcsa(torch.tensor(f, dtype=torch.float32)) for f in features_test])

# Ensure PCA applies consistently
n_pca_components = min(features_train.shape[1], 256)  # Avoid mismatch
pca = PCA(n_components=n_pca_components)
features_train = pca.fit_transform(features_train)
features_test = pca.transform(features_test)

# Scaling must be done properly
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

# Ensure `features_train_scaled` and `features_test_scaled` have the same number of features
print(f"Train Features: {features_train_scaled.shape}, Test Features: {features_test_scaled.shape}")

pca_final = PCA(n_components=features_train_scaled.shape[1])  # Avoid mismatches
features_train_reduced = pca_final.fit_transform(features_train_scaled)
features_test_reduced = pca_final.transform(features_test_scaled)


classifiers = {
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'SVM': SVC(probability=True),
    'LogisticRegression': LogisticRegression()
}

with open("model_performance.txt", "w") as file:
    for name, clf in classifiers.items():
        clf.fit(features_train_reduced, y_train)
        y_pred = clf.predict(features_test_reduced)
        y_prob = clf.predict_proba(features_test_reduced)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        file.write(f"\n{name}:\nAccuracy: {accuracy}\nF1 Score: {f1}\nROC-AUC: {roc_auc}\n")

print("Model results saved in 'model_performance.txt'.")
