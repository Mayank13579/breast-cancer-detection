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

transformer_features_train = extract_features_transformer(processor_vit, model_vit, X_train)
transformer_features_test = extract_features_transformer(processor_vit, model_vit, X_test)

# Combine features
features_train = np.concatenate((cnn_features_train, transformer_features_train), axis=1)
features_test = np.concatenate((cnn_features_test, transformer_features_test), axis=1)

# ========== Step 4: Apply PCA ==========
n_pca_components = 256
pca = PCA(n_components=n_pca_components)
features_train = pca.fit_transform(features_train)
features_test = pca.transform(features_test)

features_train = torch.tensor(features_train, dtype=torch.float32)
features_test = torch.tensor(features_test, dtype=torch.float32)

# ========== Step 5: Multi-Head Self-Attention (MHSA) ==========
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.unsqueeze(0)  # Add batch dimension
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.feature_dim ** 0.5)
        attention_probs = self.softmax(attention_scores)
        out = torch.matmul(attention_probs, v)
        return out.squeeze().cpu().numpy()

mhsa = MultiHeadSelfAttention(feature_dim=n_pca_components)
features_train = mhsa(features_train)
features_test = mhsa(features_test)

# ========== Step 6: Classification ==========
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

classifiers = {
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'SVM': SVC(probability=True),
    'LogisticRegression': LogisticRegression()
}

with open("model_performance.txt", "w") as file:
    for name, clf in classifiers.items():
        clf.fit(features_train_scaled, y_train)
        y_pred = clf.predict(features_test_scaled)
        y_prob = clf.predict_proba(features_test_scaled)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        file.write(f"\n{name}:\nAccuracy: {accuracy}\nF1 Score: {f1}\nROC-AUC: {roc_auc}\n")

print("Model results saved in 'model_performance.txt'.")
