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
    from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification
    import torch
    import torch.nn as nn
    import gc
    from transformers import AutoProcessor, AutoModel

    # Step 1: Data Preprocessing

    DATASET_PATH = './dataset'
    image_size = (224, 224)

    def load_and_preprocess_image(image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None  # Handle missing or corrupted images
        image = cv2.resize(image, image_size)
        image = image / 255.0
        return image

    # Load images and labels
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

    # Train-Test Split
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

    # Step 3: Feature Extraction - Transformer Models

    # Load ViT model
    processor_vit = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", do_rescale=False)
    model_vit = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")

    # Load Swin model
    processor_swin = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224", do_rescale=False)
    model_swin = AutoModelForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224")

    # Using batch size for low-resource PC
    def extract_features_transformer(processor, model, images, batch_size=32):
        features = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

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

        features = np.concatenate(features, axis=0)
        return features

    transformer_features_train_vit = extract_features_transformer(processor_vit, model_vit, X_train)
    transformer_features_test_vit = extract_features_transformer(processor_vit, model_vit, X_test)
    transformer_features_train_swin = extract_features_transformer(processor_swin, model_swin, X_train)
    transformer_features_test_swin = extract_features_transformer(processor_swin, model_swin, X_test)

    transformer_features_train = np.concatenate([transformer_features_train_vit, transformer_features_train_swin], axis=1)
    transformer_features_test = np.concatenate([transformer_features_test_vit, transformer_features_test_swin], axis=1)

    # Combining Features
    features_train = np.concatenate((cnn_features_train, transformer_features_train), axis=1)
    features_test = np.concatenate((cnn_features_test, transformer_features_test), axis=1)

    # Step 4: Multi-Head Self-Attention
    class MultiHeadSelfAttention(nn.Module):
        def __init__(self, embed_dim, num_heads):
            super(MultiHeadSelfAttention, self).__init__()
            self.attention = nn.MultiheadAttention(embed_dim, num_heads)

        def forward(self, x):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0).transpose(0, 1)  # Shape: (seq_len, batch_size, embed_dim)
            attn_output, _ = self.attention(x, x, x)
            return attn_output.squeeze(1).cpu().numpy()

    mhsa = MultiHeadSelfAttention(embed_dim=features_train.shape[1], num_heads=8)
    features_train = mhsa(features_train)
    features_test = mhsa(features_test)

    # Step 5: Classification
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
