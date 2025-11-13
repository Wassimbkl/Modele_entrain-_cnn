import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# -----------------------------
# 1) Chemins vers ton dataset
# -----------------------------
BASE_DIR = "data"          # adapte si besoin
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR  = os.path.join(BASE_DIR, "test")

IMG_SIZE = (48, 48)
BATCH_SIZE = 64

# -----------------------------
# 2) Data augmentation / loaders
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("Classes trouvées :", train_generator.class_indices)

# -----------------------------
# 3) Ton modèle CNN
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 émotions
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# 4) Callbacks (sauvegarde + early stopping)
# -----------------------------
checkpoint = ModelCheckpoint(
    "emotion_model_trained.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# -----------------------------
# 5) Entraînement
# -----------------------------
EPOCHS = 30

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[checkpoint, early_stop]
)

# -----------------------------
# 6) Évaluation finale
# -----------------------------
loss, acc = model.evaluate(test_generator)
print(f"\n✅ Test accuracy: {acc:.4f}, loss: {loss:.4f}")

# Sauvegarde au cas où (même si checkpoint déjà fait)
model.save("emotion_model_final.h5")

# Sauvegarde du mapping des classes
import json
with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)

print("✅ Modèle et mapping des classes sauvegardés.")
