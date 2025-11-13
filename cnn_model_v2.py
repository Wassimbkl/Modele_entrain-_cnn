from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def create_cnn_v2():
    model = Sequential([
        # Bloc 1
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(48,48,1)),
        BatchNormalization(),
        MaxPooling2D(2,2),

        # Bloc 2
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        # Bloc 3
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        # Bloc 4
        Conv2D(256, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Flatten(),

        # Dense amélioré
        Dense(256, activation='relu'),
        Dropout(0.5),

        Dense(7, activation='softmax')  # 7 émotions
    ])

    return model
