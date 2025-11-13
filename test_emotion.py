import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 1️⃣ Créer le modèle CNN (même architecture qu'Emo0.1)
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

# ⚠️ Pas encore entraîné → prédictions aléatoires

# Liste des émotions
emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# 2️⃣ Parcourir toutes les images dans le dossier "image"
image_folder = "image"  # ton dossier
for img_file in os.listdir(image_folder):
    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(image_folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48,48))
        normalized = resized / 255.0
        input_img = np.expand_dims(normalized, axis=(0,-1))

        # 3️⃣ Prédiction
        predictions = model.predict(input_img)
        emotion_index = np.argmax(predictions)

        # 4️⃣ Affichage
        print(f"{img_file} → Émotion détectée : {emotions[emotion_index]}")
