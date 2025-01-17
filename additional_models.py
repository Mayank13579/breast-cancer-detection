import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import (
    EfficientNetB3,
    DenseNet121,
    InceptionResNetV2,
    ResNet50V2,
    ConvNeXtBase
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from transformers import ViTFeatureExtractor, ViTForImageClassification, SwinForImageClassification
from tensorflow.keras.layers import Input

# Load the dataset
def load_data(dataset_path):
    images = []
    labels = []

    for label in ['IDC_negative', 'IDC_positive']:
        path = os.path.join(dataset_path, label)
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                images.append(image)
                labels.append(0 if label == 'IDC_negative' else 1)
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


# Load and preprocess dataset
dataset_path = 'dataset'  # Path to your dataset folder
images, labels = load_data(dataset_path)

# Normalize pixel values between 0 and 1
images = images / 255.0

# Encode labels
labels = to_categorical(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Utility to add custom classification layers
def add_classification_layers(base_model):
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(2, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=output)


# Function to load and compile a model
def load_model(architecture, input_shape=(224, 224, 3)):
    if architecture == "EfficientNet":
        base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_shape)
    elif architecture == "DenseNet":
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    elif architecture == "InceptionResNet":
        base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif architecture == "ResNeXt":
        base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)  # Placeholder
    elif architecture == "ConvNeXt":
        base_model = ConvNeXtBase(weights='imagenet', include_top=False, input_shape=input_shape)  # Custom import
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    # Add classification layers
    model = add_classification_layers(base_model)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Train models and evaluate
# architectures = ["EfficientNet", "DenseNet", "InceptionResNet", "ResNeXt", "ConvNeXt"]

architectures = ["ConvNeXt"]
results = {}

for arch in architectures:
    print(f"Training model: {arch}")
    model = load_model(arch)

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=10)

    # Extract features
    feature_extractor = Model(inputs=model.input, outputs=model.layers[-3].output)
    X_train_features = feature_extractor.predict(X_train)
    X_test_features = feature_extractor.predict(X_test)

    # Flatten features
    X_train_features_flat = X_train_features.reshape(X_train_features.shape[0], -1)
    X_test_features_flat = X_test_features.reshape(X_test_features.shape[0], -1)

    # Evaluate with ML classifiers
    for clf_name, clf in [("RandomForest", RandomForestClassifier()),
                          ("AdaBoost", AdaBoostClassifier()),
                          ("GradientBoosting", GradientBoostingClassifier())]:
        clf.fit(X_train_features_flat, np.argmax(y_train, axis=1))
        predictions = clf.predict(X_test_features_flat)
        accuracy = accuracy_score(np.argmax(y_test, axis=1), predictions)

        if arch not in results:
            results[arch] = {}
        results[arch][clf_name] = accuracy

    print(f"Completed training for {arch}")

# Save results to JSON
import json
with open('model_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print(json.dumps(results, indent=4))
