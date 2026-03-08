import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================
# Load Dataset
# =========================

data = np.load('./trainingDataTarget/data.npy')
target = np.load('./trainingDataTarget/target.npy')

print("Dataset Shape:", data.shape)
print("Target Shape:", target.shape)

# Normalize images
data = data / 255.0

# =========================
# Train Test Split
# =========================

train_data, test_data, train_target, test_target = train_test_split(
    data,
    target,
    test_size=0.1,
    random_state=42
)

print("Training data:", train_data.shape)
print("Testing data:", test_data.shape)

# =========================
# Data Augmentation
# =========================

datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

datagen.fit(train_data)

# =========================
# CNN Model
# =========================

model = Sequential([
    Input(shape=(32,32,1)),

    Conv2D(16,(3,3),activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(64,activation='relu'),
    Dropout(0.5),

    Dense(2,activation='softmax')
])

# =========================
# Compile Model
# =========================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# =========================
# Callbacks
# =========================

checkpoint = ModelCheckpoint(
    "best_gender_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# =========================
# Train Model
# =========================

history = model.fit(
    datagen.flow(train_data, train_target, batch_size=32),
    validation_data=(test_data, test_target),
    epochs=30,
    callbacks=[checkpoint, earlystop]
)

# =========================
# Evaluate Model
# =========================

loss, accuracy = model.evaluate(test_data, test_target)

print("\nFinal Test Accuracy:", accuracy)

# =========================
# Save Final Model
# =========================

model.save("final_gender_model.keras")

print("\nModel saved successfully!")