import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# 1. Cargar y preparar los datos
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Redimensionar para CNN [batch, alto, ancho, canales]
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# One-hot encoding de etiquetas
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Construir el modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 3. Compilar
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Entrenar
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# 5. Evaluar
loss, acc = model.evaluate(x_test, y_test)
print(f"\nPrecisión del modelo CNN en datos de prueba: {acc:.2f}")

# 6. Visualizar predicción
idx = np.random.randint(0, len(x_test))
imagen = x_test[idx]
real = y_test[idx]
pred = model.predict(imagen.reshape(1, 28, 28, 1))

plt.imshow(imagen.reshape(28, 28), cmap='gray')
plt.title(f"Predicción: {pred.argmax()} | Real: {real.argmax()}")
plt.axis('off')
plt.show()
