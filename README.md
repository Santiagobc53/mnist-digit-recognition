# MNIST Digit Recognition 🧠🔢

Este proyecto entrena y evalúa modelos de Machine Learning para reconocer dígitos escritos a mano usando el dataset MNIST. Se comparan dos arquitecturas diferentes:

- ✅ **Modelo Denso** (`mnist_modelo.py`): red neuronal simple con capas `Dense`.
- ✅ **Modelo CNN** (`mnist_cnn.py`): red neuronal convolucional, más precisa y robusta para visión por computadora.

---

## 📌 Tecnologías usadas

- Python 3.10
- TensorFlow / Keras
- Matplotlib
- NumPy

---

## 📂 Estructura del repositorio

mnist-digit-recognition/
├── mnist_modelo.py # Modelo básico con capas densas
├── mnist_cnn.py # Modelo mejorado con CNN
├── README.md # Documentación del proyecto
├── .gitignore # Archivos ignorados por Git
└── requirements.txt # Dependencias del entorno (opcional)

yaml
Copiar
Editar

---

## 🧪 Instrucciones de uso

### 1. Crear entorno virtual y activar

```bash
python -m venv tf-env
.\tf-env\Scripts\activate
2. Instalar TensorFlow y Matplotlib
bash
Copiar
Editar
pip install tensorflow matplotlib
3. Ejecutar modelo
bash
Copiar
Editar
python mnist_cnn.py
🧠 Resultados esperados
Precisión del modelo Denso: ~97–98%

Precisión del modelo CNN: ~98–99%

Visualización final con predicción del modelo

📈 Comparación de modelos
Modelo	Precisión	Arquitectura	Ideal para producción
Denso (Dense)	~0.97	Simple	❌ No
CNN	~0.98–0.99	Convolucional	✅ Sí

🚀 Próximos pasos
Añadir interfaz con Streamlit o Gradio

Implementar exportación del modelo entrenado (.h5)

Evaluar con otros datasets (Fashion-MNIST)

Autor
Santiago Barrera – LinkedIn
Transición profesional hacia el desarrollo en IA, Python y Cloud.
