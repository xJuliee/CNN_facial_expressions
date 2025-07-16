import os
import cv2
import keras
import numpy as np
import seaborn as sns
import keras_tuner as kt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from keras import layers, models, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Emotion label mapping
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
    """Loads and preprocesses grayscale images from folders mapped to class labels."""
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
                        img_equalized = cv2.equalizeHist(img_resized)  # Histogram equalization
                        img_normalized = img_equalized / 255.0
                        img_normalized = np.expand_dims(img_normalized, axis=-1)  # Add channel dimension
                        images.append(img_normalized)
                        labels.append(label)
                    else:
                        print(f"Skipping image {image_name}, size mismatch.")
                else:
                    print(f"Error reading image {image_name}")
    print(f"Loaded {len(images)} samples from {directory}")
    return np.array(images), np.array(labels)


def build_cnn(hp, input_shape=(75, 75, 1), num_classes=8):
    """Builds a CNN model using hyperparameter tuning with Keras Tuner."""
    model = models.Sequential()
    model.add(Input(shape=input_shape))

    # Convolution Block 1
    filters_1 = hp.Choice('filters_1', [16, 32])
    model.add(layers.Conv2D(filters_1, (7, 7), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters_1, (7, 7), padding='same', activation='relu'))
    model.add(layers.AveragePooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(hp.Float('dropout_1', 0.3, 0.6, step=0.1)))

    # Convolution Block 2
    filters_2 = hp.Choice('filters_2', [32, 64])
    model.add(layers.Conv2D(filters_2, (5, 5), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters_2, (5, 5), padding='same', activation='relu'))
    model.add(layers.AveragePooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(hp.Float('dropout_2', 0.3, 0.6, step=0.1)))

    # Convolution Block 3
    filters_3 = hp.Choice('filters_3', [64, 128])
    model.add(layers.Conv2D(filters_3, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters_3, (3, 3), padding='same', activation='relu'))
    model.add(layers.AveragePooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(hp.Float('dropout_3', 0.3, 0.6, step=0.1)))

    # Convolution Block 4
    filters_4 = hp.Choice('filters_4', [128, 256])
    model.add(layers.Conv2D(filters_4, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters_4, (3, 3), padding='same', activation='relu'))
    model.add(layers.AveragePooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(hp.Float('dropout_4', 0.3, 0.6, step=0.1)))

    # Final Convolution Layer (no activation on last conv)
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(num_classes, (3, 3), padding='same'))

    # Global Average Pooling followed by Softmax
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Activation('softmax'))

    # Compile with tunable learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def tune_cnn_hyperparameters(build_fn, X_train, y_train, X_val, y_val, max_epochs=3, factor=4):
    """Tunes hyperparameters using Keras Tuner's Hyperband algorithm."""
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
    """Plots training history and evaluates model on the test set."""
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.title('Loss over epochs')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.legend()
    plt.title('Accuracy over epochs')
    plt.show()

    # Evaluate and report
    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    tick_labels = list(class_mapping.keys())

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=tick_labels, yticklabels=tick_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=tick_labels))


if __name__ == "__main__":
    print("Loading data from directories...")

    train_path = "FER_data/train"
    val_path = "FER_data/val"
    test_path = "FER_data/test"

    # Load datasets
    X_train, y_train = load_data_from_dir(train_path)
    X_val, y_val = load_data_from_dir(val_path)
    X_test, y_test = load_data_from_dir(test_path)

    print("Tuning hyperparameters...")
    best_model, best_params = tune_cnn_hyperparameters(build_cnn, X_train, y_train, X_val, y_val)

    print("Training best model...")

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    checkpoint = ModelCheckpoint("CNNs/BEST_CNN.keras", monitor='val_accuracy', save_best_only=True, verbose=1)

    history = best_model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[reduce_lr, checkpoint]
    )

    plots_and_evaluation(history, best_model, X_test, y_test)

    best_model.save("New_Model.keras")
    print("Model saved.")