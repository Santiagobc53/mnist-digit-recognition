import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. Cargar dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalizar los datos (0-255 → 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Codificar etiquetas (one-hot)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 4. Crear modelo
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 5. Compilar modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. Entrenar modelo
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# 7. Evaluar en test
loss, acc = model.evaluate(x_test, y_test)
print(f"\nPrecisión en datos de prueba: {acc:.2f}")

# 8. Visualizar una predicción
import numpy as np
idx = np.random.randint(0, len(x_test))
imagen = x_test[idx]
real = y_test[idx]
pred = model.predict(imagen.reshape(1, 28, 28))

plt.imshow(imagen, cmap='gray')
plt.title(f"Predicción: {pred.argmax()} | Real: {real.argmax()}")
plt.axis('off')
plt.show()
