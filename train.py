# 1️⃣ Importer les modules
import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------------
# 2️⃣ Préparer les images pour l'entraînement
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(48,48),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(48,48),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# -------------------------------
# 3️⃣ Créer le modèle CNN
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

# -------------------------------
# 4️⃣ Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------------------
# 5️⃣ Entraîner le modèle
model.fit(
    train_generator,
    epochs=10,  # tu peux augmenter pour un meilleur entraînement
    validation_data=validation_generator
)

# -------------------------------
# 6️⃣ Sauvegarder le modèle entraîné
model.save("emotion_model_trained.h5")

# -------------------------------
# 7️⃣ Tester les images du dossier "image"
emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
image_folder = "image"

# Charger le modèle sauvegardé (optionnel si tu veux juste tester après entraînement)
# model = load_model("emotion_model_trained.h5")

for img_file in os.listdir(image_folder):
    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(image_folder, img_file)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48,48))
        normalized = resized / 255.0
        input_img = np.expand_dims(normalized, axis=(0,-1))
        
        prediction = model.predict(input_img)
        emotion_index = np.argmax(prediction)
        print(f"{img_file} → Émotion détectée : {emotions[emotion_index]}")
