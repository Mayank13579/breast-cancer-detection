import ssl
import urllib3

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from tensorflow.keras.applications import ResNet50, InceptionV3, VGG16
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import torch.nn as nn
import gc

# Step 1: Data Preprocessing
DATASET_PATH = './dataset/IDC/training'
image_size = (224, 224)

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, image_size)
    image = image / 255.0
    return image

X, y = [], []
for label in ['0', '1']:
    folder_path = os.path.join(DATASET_PATH, label)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        image = load_and_preprocess_image(img_path)
        if image is not None:
            X.append(image)
            y.append(0 if label == '0' else 1)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Feature Extraction - CNN Models

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

# Step 3: Feature Fusion - Linear Attention
class LinearAttention(nn.Module):
    def __init__(self, embed_dim):
        super(LinearAttention, self).__init__()
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = self.linear(x)
        return x.detach().cpu().numpy()

linear_attention = LinearAttention(embed_dim=cnn_features_train.shape[1])
features_train = linear_attention(cnn_features_train)
features_test = linear_attention(cnn_features_test)

# Step 4: Classification
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

pca = PCA(n_components=512)
features_train_reduced = pca.fit_transform(features_train_scaled)
features_test_reduced = pca.transform(features_test_scaled)

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
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        file.write(f'\n{name} Classifier:\n')
        file.write(f'Accuracy: {accuracy}\n')
        file.write(f'Precision: {precision}\n')
        file.write(f'Recall: {recall}\n')
        file.write(f'F1 Score: {f1}\n')
        file.write(f'ROC-AUC: {roc_auc}\n')
        file.write(f'Confusion Matrix:\n{conf_matrix}\n')
        file.write(f'Classification Report:\n{class_report}\n')

print("\nModel performance has been saved to 'model_performance.txt'.")
