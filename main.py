

"""
PARTIE 1 : TEST RAPIDE AVEC DEEPFACE (À FAIRE EN PREMIER)
----------------------------------------------------------
Testez immédiatement si le modèle fonctionne sur vos images
"""


import os
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

# ============= TEST 1 : Analyse d'une seule image =============
def test_single_image(image_path):
    """
    Teste DeepFace sur une seule image
    """
    print(f"🔍 Analyse de l'image : {image_path}")
    
    try:
        # Analyse l'image
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion'],
            enforce_detection=False  # Continue même si visage pas détecté
        )
        
        print("\n✅ Résultats :")
        print(f"Émotion dominante : {result[0]['dominant_emotion']}")
        print(f"\nToutes les émotions détectées :")
        for emotion, score in result[0]['emotion'].items():
            print(f"  - {emotion}: {score:.2f}%")
        
        return result
    
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return None


