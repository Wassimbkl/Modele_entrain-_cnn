model.compile(
    optimizer='adam',               # 1️⃣ Optimiseur
    loss='categorical_crossentropy',# 2️⃣ Fonction de perte
    metrics=['accuracy']            # 3️⃣ Métrique
)
