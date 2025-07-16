import os
import cv2
import keras
import numpy as np
import seaborn as sns
import keras_tuner as kt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras import layers, models, Input
from keras.callbacks import EarlyStopping

# Dataset path
dataset_path = "FER_data/train"

# Updated 8-class mapping (shy removed)
class_mapping = {
    "angry": 0,
    "confused": 1,
    "disgust": 2,
    "fear": 3,
    "happy": 4,
    "neutral": 5,
    "sad": 6,
    "surprise": 7
}

def load_data_from_dir(directory, image_size=(75, 75)):
    images, labels = [], []

    for folder_name, label in class_mapping.items():
        class_dir = os.path.join(directory, folder_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Folder {folder_name} not found in {directory}")
            continue

        for image_name in os.listdir(class_dir):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(class_dir, image_name)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, image_size)
                    if img_resized.shape == image_size:
                        img_equalized = cv2.equalizeHist(img_resized)
                        img_normalized = img_equalized / 255.0
                        img_normalized = np.expand_dims(img_normalized, axis=-1)
                        images.append(img_normalized)
                        labels.append(label)
                    else:
                        print(f"Skipping image {image_name}, size mismatch.")
                else:
                    print(f"Error reading image {image_name}")

    print(f"Loaded {len(images)} samples from {directory}")
    return np.array(images), np.array(labels)


def build_cnn(hp, input_shape=(75, 75, 1), num_classes=8):
    model = models.Sequential([
        Input(shape=input_shape),
        layers.Conv2D(hp.Int('num_filters_1', 32, 128, step=32), (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(hp.Int('num_filters_2', 64, 128, step=32), (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(hp.Int('num_filters_3', 128, 256, step=64), (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(hp.Float('dropout_rate', 0.3, 0.7, step=0.1)),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def tune_cnn_hyperparameters(build_fn, X_train, y_train, X_val, y_val, max_epochs=3, factor=4):
    tuner = kt.Hyperband(
        build_fn,
        objective='val_accuracy',
        max_epochs=max_epochs,
        factor=factor,
        directory='cnn_tuning',
        project_name='emotion_classification',
        hyperband_iterations=2
    )

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)
    tuner.search(X_train, y_train, epochs=max_epochs, validation_data=(X_val, y_val), callbacks=[early_stopping])
    best_model = tuner.get_best_models(1)[0]
    best_params = tuner.get_best_hyperparameters(1)[0]
    print("Best Hyperparameters:", best_params.values)
    return best_model, best_params

def plots_and_evaluation(history, model, X_test, y_test):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.title('Loss over epochs')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.legend()
    plt.title('Accuracy over epochs')
    plt.show()

    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    tick_labels = list(class_mapping.keys())

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=tick_labels, yticklabels=tick_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=tick_labels))

if __name__ == "__main__":
    print("Loading data from directories...")

    train_path = "FER_data/train"
    val_path   = "FER_data/val"
    test_path  = "FER_data/test"

    X_train, y_train = load_data_from_dir(train_path)
    X_val, y_val     = load_data_from_dir(val_path)
    X_test, y_test   = load_data_from_dir(test_path)

    print("Tuning hyperparameters...")
    best_model, best_params = tune_cnn_hyperparameters(build_cnn, X_train, y_train, X_val, y_val)

    print("Training best model...")
    history = best_model.fit(X_train, y_train, epochs=24, batch_size=32, validation_data=(X_val, y_val))

    plots_and_evaluation(history, best_model, X_test, y_test)

    best_model.save("NEW_emotion_cnn_best_model.keras")
    print("Model saved.")