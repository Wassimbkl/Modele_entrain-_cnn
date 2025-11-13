from fer import FER
import cv2

detector = FER(mtcnn=True)

# mets n’importe quelle image ici
img = cv2.imread("image/test.jpg")

if img is None:
    raise Exception("Image introuvable ! Place une image dans /images/test.jpg")

results = detector.detect_emotions(img)

print("Résultats :", results)

# Afficher l'image avec les émotions détectées
for face in results:
    (x, y, w, h) = face["box"]
    emotion = max(face["emotions"], key=face["emotions"].get)

    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(img, emotion, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

cv2.imshow("Test Emotion", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
