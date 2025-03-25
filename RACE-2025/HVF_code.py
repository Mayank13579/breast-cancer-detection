import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import timm  # For Vision Transformers
import numpy as np
import cv2
import os
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

print("Starting data preprocessing...")
# Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_path = "dataset/training/"  # Path to the folder containing IDC_positive and IDC_negative
dataset = datasets.ImageFolder(root=data_path, transform=transform)
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

print("Loading feature extraction models...")
# Feature Extraction Models
cnn_models = {
    "resnet50": models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
    "vgg19": models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1),
    "efficientnet": models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
}

transformer_models = {
    "vit": timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0),
    "swin": timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)
}

def extract_features(model, dataloader):
    print(f"Extracting features using {model.__class__.__name__}...")
    model = model.to(device)
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, label in dataloader:
            images = images.to(device)
            feat = model(images).cpu().numpy()
            features.extend(feat)
            labels.extend(label.numpy())
    return np.array(features), np.array(labels)

# Extract Features
cnn_features, labels = extract_features(cnn_models["resnet50"], train_loader)
vit_features, _ = extract_features(transformer_models["vit"], train_loader)

print("Performing feature fusion...")
# Cross-Modal Fusion (Concatenation + PCA Reduction)
from sklearn.decomposition import PCA
fusion_features = np.concatenate((cnn_features, vit_features), axis=1)
pca = PCA(n_components=256)
fusion_features = pca.fit_transform(fusion_features)

print("Training classifiers...")
# Train Stacking Classifier
rf = RandomForestClassifier(n_estimators=100)
gb = GradientBoostingClassifier(n_estimators=100)
svm = SVC(probability=True)

rf.fit(fusion_features, labels)
gb.fit(fusion_features, labels)
svm.fit(fusion_features, labels)

print("Making predictions...")
# Ensemble Model Prediction
rf_preds = rf.predict(fusion_features)
gb_preds = gb.predict(fusion_features)
svm_preds = svm.predict(fusion_features)

final_preds = np.round((rf_preds + gb_preds + svm_preds) / 3)

# Evaluate Model
print("Classification Report:")
print(classification_report(labels, final_preds))

# Compute Confusion Matrix
conf_matrix = confusion_matrix(labels, final_preds)
print("Confusion Matrix:")
print(conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Compute AUC-ROC Score
roc_auc = roc_auc_score(labels, final_preds)
print(f"AUC-ROC Score: {roc_auc:.4f}")

# Compute Accuracy, Precision, Recall
accuracy = accuracy_score(labels, final_preds)
print(f"Accuracy: {accuracy:.4f}")
