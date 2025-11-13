import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model

# -------------------------
# 1) Charger ton modèle
# -------------------------
model = load_model("emotion_model_trained.h5")

# -------------------------
# 2) Charger le mapping des classes
# -------------------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Inverser le dictionnaire : index -> nom classe
index_to_class = {v: k for k, v in class_indices.items()}

# -------------------------
# 3) Chemin de l'image
# -------------------------
IMAGE_PATH = "image/PrivateTest_1290484.jpg"     # mets ton image ici

# Charger l'image
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise Exception("❌ Impossible de charger l'image ! Vérifie le chemin.")

# Convertir en niveaux de gris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Redimensionner en 48x48
resized = cv2.resize(gray, (48, 48))

# Normaliser et reshape pour le réseau
input_img = resized.reshape(1, 48, 48, 1) / 255.0

# -------------------------
# 4) Prédiction
# -------------------------
pred = model.predict(input_img)[0]      # tableau de 7 probabilités
emotion_index = np.argmax(pred)
emotion = index_to_class[emotion_index]
confidence = pred[emotion_index]

print("\n🎯 Emotion détectée :", emotion)
print(f"🔍 Confiance : {confidence:.2f}")

# -------------------------
# 5) Affichage sur l'image
# -------------------------
cv2.putText(
    img,
    f"{emotion} ({confidence:.2f})",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.imshow("Prediction CNN", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
