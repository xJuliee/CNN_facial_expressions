import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, BatchNormalization, Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === Shared class names to enforce consistent label mapping ===
class_names = ['angry', 'confused', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'shy', 'surprise']
class_indices = {name: i for i, name in enumerate(class_names)}
print("Class indices:", class_indices)

n_classes = len(class_names)  # <<< define before class_weights!

# === Load datasets with integer labels ===
def prepare_dataset(folder_path, batch_size=256):
    ds = tf.keras.utils.image_dataset_from_directory(
        folder_path,
        labels="inferred",
        label_mode="int",                # <<< integer labels
        image_size=(75, 75),
        color_mode="grayscale",
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        class_names=class_names
    )
    # Normalize pixel values
    return ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)) \
             .prefetch(tf.data.AUTOTUNE)

print("Loading datasets...")
train_ds = prepare_dataset("FER_data/train")
val_ds   = prepare_dataset("FER_data/val")
test_ds  = prepare_dataset("FER_data/test")

# === Compute class weights only for 'confused' and 'shy' ===
sample_counts = {
    'angry':   6472,
    'confused': 891,
    'disgust': 6472,
    'fear':    6472,
    'happy':   6472,
    'neutral': 6472,
    'sad':     6472,
    'shy':     691,
    'surprise':6472
}
max_count = max(sample_counts.values())

# Make sure class_weights covers all classes, default=1.0
class_weights = {i: 1.0 for i in range(n_classes)}
class_weights[class_indices['confused']] = max_count / sample_counts['confused']
class_weights[class_indices['shy']] = max_count / sample_counts['shy']
print("Applied class weights:", class_weights)

# === CNN model definition ===
def build_cnn(input_shape, n_classes):
    model = Sequential([
        Input(shape=input_shape),
        BatchNormalization(),

        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.25),

        Flatten(),
        Dense(512),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.25),

        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.25),

        Dense(n_classes, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',  # <<< integer label loss
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

# === Training setup ===
batch_size = 256
epochs     = 3
input_shape= (75, 75, 1)

print("Training CNN...")
model = build_cnn(input_shape, n_classes)
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    ModelCheckpoint("saved_model/cnn_model_75x75.h5", save_best_only=True, monitor='val_accuracy', mode='max')
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=callbacks
)

# === Evaluate on test set ===
print("Evaluating model...")
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# Build y_true from test_ds
y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)

# === Confusion matrix ===
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# === Classification report ===
print("Classification Report:\n" +
      classification_report(y_true, y_pred, target_names=class_names))
