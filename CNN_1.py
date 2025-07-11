import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Load and preprocess dataset (grayscale, 48x48) ===
dataset = tf.keras.utils.image_dataset_from_directory(
    "fane_data_resized_48x48_gray",  # folder with 48x48 grayscale images
    labels="inferred",
    label_mode="int",
    image_size=(48, 48),
    batch_size=None,
    color_mode="grayscale",  # <- grayscale loading
    shuffle=True,
    seed=42
)

images, labels = [], []
for img, label in dataset:
    images.append(img.numpy())
    labels.append(label.numpy())

X = np.array(images)
y = np.array(labels)

# Normalize images
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

n_classes = len(np.unique(y))
Y_train = to_categorical(y_train, n_classes)
Y_test = to_categorical(y_test, n_classes)

# === Flip images horizontally for ensemble strategy ===
X_train_flipped = np.flip(X_train, axis=2)
X_test_flipped = np.flip(X_test, axis=2)

# === CNN architecture adjusted for 48x48 grayscale images ===
def build_cnn(input_shape, n_classes):
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# === Training setup ===
batch_size = 128
epochs = 5
input_shape = X_train.shape[1:]  # should be (48, 48, 1)

# === Compute class weights ===
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights_array))

# === Train CNNs ===
print("Training model on original images...")
model_orig = build_cnn(input_shape, n_classes)
model_orig.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, class_weight=class_weights)

print("Training model on flipped images...")
model_flipped = build_cnn(input_shape, n_classes)
model_flipped.fit(X_train_flipped, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, class_weight=class_weights)

# === Save CNN models ===
os.makedirs('saved_model', exist_ok=True)
model_orig.save('saved_model/cnn0-2.h5')
model_flipped.save('saved_model/cnn1-2.h5')

# === Ensemble predictions ===
pred_train_orig = model_orig.predict(X_train)
pred_test_orig = model_orig.predict(X_test)

pred_train_flipped = model_flipped.predict(X_train_flipped)
pred_test_flipped = model_flipped.predict(X_test_flipped)

# Concatenate predictions for MLP input
p_train = np.concatenate((pred_train_orig, pred_train_flipped), axis=1)
p_test = np.concatenate((pred_test_orig, pred_test_flipped), axis=1)

# === MLP ensemble ===
ensemble = Sequential()
ensemble.add(Dense(128, activation='relu', input_shape=(p_train.shape[1],)))
ensemble.add(Dense(n_classes, activation='softmax'))

ensemble.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# === Callbacks ===
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

h5_path = "saved_model/ensemble_model-2.h5"
keras_path = "saved_model/ensemble_model-2.keras"
checkpoint_h5 = ModelCheckpoint(h5_path, save_best_only=True, monitor='val_accuracy', mode='max')
checkpoint_keras = ModelCheckpoint(keras_path, save_best_only=True, monitor='val_accuracy', mode='max')

print("Training ensemble model on predictions...")
ensemble.fit(
    p_train, Y_train,
    batch_size=32,
    epochs=20,
    validation_data=(p_test, Y_test),
    callbacks=[early_stop, checkpoint_h5, checkpoint_keras]
)

# Save ensemble model
ensemble.save('saved_model/ensemble_model-2.h5')

# === Evaluation ===
y_pred_probs = ensemble.predict(p_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(Y_test, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
class_names = dataset.class_names

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:\n", report)
