import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model

# -------------------------
# 1) Charger le modèle V2
# -------------------------
model = load_model("emotion_model_v2.h5")

# -------------------------
# 2) Charger le mapping des classes
# -------------------------
with open("class_indices_v2.json", "r") as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}

# -------------------------
# 3) Image à tester
# -------------------------
IMAGE_PATH = "image/test.jpg"  # mets ton image ici

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise Exception("❌ Impossible de charger l'image. Vérifie le chemin.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (48, 48))
input_img = resized.reshape(1, 48, 48, 1) / 255.0

# -------------------------
# 4) Prédiction
# -------------------------
pred = model.predict(input_img)[0]
emotion_index = np.argmax(pred)
emotion = index_to_class[emotion_index]
confidence = pred[emotion_index]

print("\n🎯 Emotion détectée :", emotion)
print(f"🔍 Confiance : {confidence:.2f}")

# -------------------------
# 5) Affichage
# -------------------------
cv2.putText(img, f"{emotion} ({confidence:.2f})",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0,255,0), 2)

cv2.imshow("Emotion Prediction V2", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
