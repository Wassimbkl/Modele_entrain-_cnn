import cv2
from fer import FER
import pandas as pd
from datetime import datetime

# Initialisation du détecteur FER
detector = FER(mtcnn=True)

# Fichier image à analyser
IMAGE_PATH = "image/test.jpg"   # <-- Mets ici le chemin de ton image

# Charger l'image
img = cv2.imread(IMAGE_PATH)

if img is None:
    raise Exception("❌ Impossible de charger l'image : vérifie le chemin.")

# Détection des émotions
results = detector.detect_emotions(img)

if len(results) == 0:
    print("😕 Aucun visage détecté.")
else:
    for face in results:
        (x, y, w, h) = face["box"]
        emotions = face["emotions"]

        # Émotion dominante
        top_emotion = max(emotions, key=emotions.get)
        confidence = emotions[top_emotion]

        # Rectangle sur le visage
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Texte émotion + confiance
        text = f"{top_emotion} ({confidence:.2f})"
        cv2.putText(img, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # BONUS CSV
        log = pd.DataFrame([[datetime.now(), top_emotion, confidence]],
                           columns=["timestamp", "emotion", "confidence"])
        log.to_csv("emotions_log.csv", mode="a", header=False, index=False)

# Afficher l'image finale
cv2.imshow("Emotion Detection - Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
