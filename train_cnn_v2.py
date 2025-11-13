import json
from cnn_model_v2 import create_cnn_v2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -------------------------
# 1 - Data Augmentation
# -------------------------
train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    "data/train",
    target_size=(48,48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    "data/test",
    target_size=(48,48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

# -------------------------
# 2 - Charger CNN V2
# -------------------------
model = create_cnn_v2()

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------
# 3 - Callbacks
# -------------------------
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True),
    ModelCheckpoint("emotion_model_v2.h5", save_best_only=True)
]

# -------------------------
# 4 - Entraînement V2
# -------------------------
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=35,
    callbacks=callbacks
)

# -------------------------
# 5 - Sauvegarde du mapping
# -------------------------
with open("class_indices_v2.json", "w") as f:
    json.dump(train_generator.class_indices, f)

print("🎉 Modèle V2 entraîné et sauvegardé.")
