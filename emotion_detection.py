import cv2
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
import pandas as pd

# Charger le modèle
model = load_model("emotion_model_trained.h5")

# Classes d'émotions (à adapter selon ton modèle)
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

# CSV Log
csv_file = "emotions.csv"
df = pd.DataFrame(columns=["timestamp", "emotion"])
df.to_csv(csv_file, index=False)

print("🎥 Détection démarrée — appuie sur Q pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype("float32") / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        predictions = model.predict(face)
        emotion = EMOTIONS[np.argmax(predictions)]

        # Dessin
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # Sauvegarde CSV
        new_row = pd.DataFrame([[datetime.now(), emotion]], columns=["timestamp", "emotion"])
        new_row.to_csv(csv_file, mode='a', header=False, index=False)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
