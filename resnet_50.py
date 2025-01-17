import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


# Load the dataset
# Assuming dataset is in the following structure:
# - dataset/
#     - IDC_negative/
#         - image_1.png
#         - image_2.png
#     - IDC_positive/
#         - image_1.png
#         - image_2.png

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
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create augmented data generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
test_generator = ImageDataGenerator().flow(X_test, y_test, batch_size=32)




from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load the ResNet50V2 model with pre-trained weights
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers (optional)
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for feature extraction
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Dropout to prevent overfitting
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(2, activation='softmax')(x)

# Define the model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()


# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator
)

# Save the trained model
model.save('breast_cancer_detection_model.h5')



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Extract deep features using the trained ResNet50V2-based model
def extract_features(model, data):
    feature_extractor = Model(inputs=model.input, outputs=model.layers[-3].output)  # Using the second last layer
    features = feature_extractor.predict(data)
    return features

# Extract features from training and testing datasets
X_train_features = extract_features(model, X_train)
X_test_features = extract_features(model, X_test)

# Flatten the features for use in traditional ML classifiers
X_train_features_flat = X_train_features.reshape(X_train_features.shape[0], -1)
X_test_features_flat = X_test_features.reshape(X_test_features.shape[0], -1)


# Initialize machine learning classifiers
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
ada_classifier = AdaBoostClassifier(n_estimators=100, random_state=42)
gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Train classifiers on extracted features
rf_classifier.fit(X_train_features_flat, np.argmax(y_train, axis=1))
ada_classifier.fit(X_train_features_flat, np.argmax(y_train, axis=1))
gb_classifier.fit(X_train_features_flat, np.argmax(y_train, axis=1))

# Evaluate classifiers on test set
rf_predictions = rf_classifier.predict(X_test_features_flat)
ada_predictions = ada_classifier.predict(X_test_features_flat)
gb_predictions = gb_classifier.predict(X_test_features_flat)

# Print evaluation metrics
print("Random Forest Classifier:")
print(f"Accuracy: {accuracy_score(np.argmax(y_test, axis=1), rf_predictions)}")
print(classification_report(np.argmax(y_test, axis=1), rf_predictions))

print("AdaBoost Classifier:")
print(f"Accuracy: {accuracy_score(np.argmax(y_test, axis=1), ada_predictions)}")
print(classification_report(np.argmax(y_test, axis=1), ada_predictions))

print("Gradient Boosting Classifier:")
print(f"Accuracy: {accuracy_score(np.argmax(y_test, axis=1), gb_predictions)}")
print(classification_report(np.argmax(y_test, axis=1), gb_predictions))


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Perform 10-fold cross-validation
def cross_validation(classifier, X, y, k=10):
    skf = StratifiedKFold(n_splits=k)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    roc_aucs = []

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Train classifier
        classifier.fit(X_train, y_train)
        
        # Predict on validation set
        predictions = classifier.predict(X_val)
        probabilities = classifier.predict_proba(X_val)[:, 1]
        
        # Evaluation metrics
        accuracies.append(accuracy_score(y_val, predictions))
        precisions.append(precision_score(y_val, predictions))
        recalls.append(recall_score(y_val, predictions))
        f1_scores.append(f1_score(y_val, predictions))
        roc_aucs.append(roc_auc_score(y_val, probabilities))

    return {
        "accuracy": np.mean(accuracies),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1_score": np.mean(f1_scores),
        "roc_auc": np.mean(roc_aucs)
    }

# Run cross-validation for each classifier
rf_cv_results = cross_validation(rf_classifier, X_train_features_flat, np.argmax(y_train, axis=1))
ada_cv_results = cross_validation(ada_classifier, X_train_features_flat, np.argmax(y_train, axis=1))
gb_cv_results = cross_validation(gb_classifier, X_train_features_flat, np.argmax(y_train, axis=1))

# Print cross-validation results
print("Cross-Validation Results:")
print("Random Forest Classifier:", rf_cv_results)
print("AdaBoost Classifier:", ada_cv_results)
print("Gradient Boosting Classifier:", gb_cv_results)



# Plot training and validation accuracy/loss
def plot_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))
    
    # Plot Accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

# Plot ROC Curve for a given classifier
def plot_roc_curve(classifier, X_test, y_test):
    probabilities = classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(np.argmax(y_test, axis=1), probabilities)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

# Visualize the training results
plot_training(history)

# Plot ROC curves for each classifier
print("ROC Curve - Random Forest:")
plot_roc_curve(rf_classifier, X_test_features_flat, y_test)

print("ROC Curve - AdaBoost:")
plot_roc_curve(ada_classifier, X_test_features_flat, y_test)

print("ROC Curve - Gradient Boosting:")
plot_roc_curve(gb_classifier, X_test_features_flat, y_test)


import json

# Save results to a JSON file
results = {
    "Random Forest": {
        "cross_validation": rf_cv_results,
        "test_accuracy": accuracy_score(np.argmax(y_test, axis=1), rf_predictions)
    },
    "AdaBoost": {
        "cross_validation": ada_cv_results,
        "test_accuracy": accuracy_score(np.argmax(y_test, axis=1), ada_predictions)
    },
    "Gradient Boosting": {
        "cross_validation": gb_cv_results,
        "test_accuracy": accuracy_score(np.argmax(y_test, axis=1), gb_predictions)
    }
}

# Save results to JSON file
with open('model_performance_report.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)

# Print the performance summary
print("Performance Summary:")
print(json.dumps(results, indent=4))
