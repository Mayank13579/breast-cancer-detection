# Novel Hybrid Vision Fusion (NHVF) Implementation

import numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingClassifier, VotingClassifier

import tensorflow as tf
from tensorflow.keras.applications import ResNet50, InceptionV3, DenseNet121
from transformers import ViTFeatureExtractor, ViTModel, SwinModel

# 1. Data Preprocessing
print('Loading and Preprocessing Data...')
def load_data(data_dir):
    X, y = [], []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (224, 224)) / 255.0
            X.append(image)
            y.append(label)
    return np.array(X), np.array(y)

X, y = load_data('/dataset')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Feature Extraction with CNN Models
print('Extracting Features using CNNs...')
cnn_models = [ResNet50(weights='imagenet', include_top=False),
              InceptionV3(weights='imagenet', include_top=False),
              DenseNet121(weights='imagenet', include_top=False)]

cnn_features_train, cnn_features_test = [], []
for model in cnn_models:
    train_features = model.predict(X_train)
    test_features = model.predict(X_test)
    train_features = train_features.reshape(train_features.shape[0], -1)
    test_features = test_features.reshape(test_features.shape[0], -1)
    cnn_features_train.append(train_features)
    cnn_features_test.append(test_features)

cnn_features_train = np.hstack(cnn_features_train)
cnn_features_test = np.hstack(cnn_features_test)

# 3. Feature Extraction with Transformer Models
print('Extracting Features using Transformers...')
transformer_models = [ViTModel.from_pretrained('google/vit-base-patch16-224-in21k'),
                      SwinModel.from_pretrained('microsoft/swin-base-patch4-window7-224')]

transformer_features_train, transformer_features_test = [], []
for model in transformer_models:
    train_features = model(X_train)['last_hidden_state'].detach().numpy()
    test_features = model(X_test)['last_hidden_state'].detach().numpy()
    train_features = train_features.reshape(train_features.shape[0], -1)
    test_features = test_features.reshape(test_features.shape[0], -1)
    transformer_features_train.append(train_features)
    transformer_features_test.append(test_features)

transformer_features_train = np.hstack(transformer_features_train)
transformer_features_test = np.hstack(transformer_features_test)

# 4. Feature Fusion
print('Fusing Features...')
features_train = np.hstack([cnn_features_train, transformer_features_train])
features_test = np.hstack([cnn_features_test, transformer_features_test])

# Optional Dimensionality Reduction (PCA)
pca = PCA(n_components=512)
features_train = pca.fit_transform(features_train)
features_test = pca.transform(features_test)

# 5. Classification using Stacking
print('Training Stacking Classifier...')
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('gb', GradientBoostingClassifier(n_estimators=100)),
    ('svc', SVC(kernel='linear', probability=True))
]
stacking_clf = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())
stacking_clf.fit(features_train, y_train)

# 6. Evaluation
print('Evaluating Model...')
y_pred = stacking_clf.predict(features_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

# 7. Inference
print('Performing Inference on New Data...')
def inference(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224)) / 255.0
    features = np.hstack([model.predict(image[np.newaxis, ...]) for model in cnn_models])
    transformer_features = np.hstack([model(image[np.newaxis, ...])['last_hidden_state'].detach().numpy() for model in transformer_models])
    features = np.hstack([features, transformer_features])
    features = pca.transform(features)
    prediction = stacking_clf.predict(features)
    return prediction

print('NHVF Pipeline Complete!')