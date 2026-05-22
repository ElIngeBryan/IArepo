import tensorflow as tf
import numpy as np
import cv2
import gradio as gr
import os

# Ocultar advertencias de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. Configuración inicial
CLASES = ['Arañas', 'Ballenas', 'Changos', 'Pájaros', 'Ranas']
RUTA_MODELO = "../modelos/modelo_animales.keras"

print("Cargando el cerebro artificial...")
try:
    modelo = tf.keras.models.load_model(RUTA_MODELO)
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# 2. La función de predicción (El núcleo de la IA)
def clasificar_animal(imagen):
    # Gradio ya nos entrega la imagen lista como un arreglo de NumPy en RGB
    if imagen is None:
        return {"Sube una imagen": 0.0}

    # Redimensionamos a 96x96 (como lo configuraste en tu Script 3)
    img_redimensionada = cv2.resize(imagen, (96, 96))
    
    # Preparamos la imagen para Keras agregando la dimensión del lote: (1, 96, 96, 3)
    img_array = np.expand_dims(img_redimensionada, axis=0)

    # Inferencia
    predicciones = modelo.predict(img_array, verbose=0)[0]

    # Formateamos el resultado como un diccionario { "Nombre Animal": Probabilidad }
    # Esto permite que la interfaz web dibuje barras de progreso automáticamente
    resultados = {CLASES[i]: float(predicciones[i]) for i in range(len(CLASES))}
    
    return resultados

# 3. Construcción de la Interfaz Gráfica con Gradio
print("¡Iniciando Servidor Web!")

interfaz = gr.Interface(
    fn=clasificar_animal,                # La función que procesa la imagen
    inputs=gr.Image(),                   # Componente de entrada
    outputs=gr.Label(num_top_classes=3), # Componente de salida
    title="Clasificador de Animales con IA",
    description="",
    flagging_mode="never"                # Versión actualizada para ocultar el botón de reporte
)

# 4. Lanzar la aplicación
if __name__ == "__main__":
    # El tema ahora se pasa aquí adentro en las nuevas versiones
    interfaz.launch(share=False, theme="soft")