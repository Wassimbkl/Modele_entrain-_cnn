Emotion Detection using CNN (Deep Learning)

Ce projet implémente un modèle de Deep Learning capable de reconnaître les émotions humaines à partir d’images de visages. 
Le modèle utilise un réseau de neurones convolutionnel (CNN) entraîné sur un dataset d’images en niveaux de gris (48×48).

---

Fonctionnalités

- Détection d’émotions à partir d’une image (`predict_image_v2.py`)
- Modèle entraîné avec un CNN amélioré (V2)
- 7 classes d’émotions :
  - angry
  - disgust
  - fear
  - happy
  - neutral
  - sad
  - surprise



 Architecture du Modèle (CNN V2)

Le modèle utilise :

- 4 blocs Convolution + MaxPooling
- Batch Normalization
- Dense Layer (256 neurones)
- Dropout (0.5)
- Softmax pour les 7 classes

Fichiers:

- cnn_model_v2.py → architecture du modèle  
- train_cnn_v2.py → script d’entraînement  
- predict_image_v2.py → script pour tester une image  
- class_indices_v2.json → mapping des classes

 

---

Installation
