import cv2
import numpy as np
import tensorflow as tf

# 1. Cargar el modelo entrenado
# Nota: Asegúrate de que el orden de las clases coincida con el que imprimió el script de entrenamiento
CLASES = ['aranas', 'ballenas', 'changos', 'pajaros', 'ranas'] 
modelo = tf.keras.models.load_model("../modelos/modelo_animales.keras")

# 2. Inicializar la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se puede abrir la cámara.")
    exit()

print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Obtener dimensiones del frame
    alto, ancho, _ = frame.shape

    # 3. Crear el "Bounding Box" en el centro de la pantalla
    # Vamos a crear un cuadrado de 300x300 en el centro
    tamano_caja = 300
    x_inicio = int((ancho - tamano_caja) / 2)
    y_inicio = int((alto - tamano_caja) / 2)
    x_fin = x_inicio + tamano_caja
    y_fin = y_inicio + tamano_caja

    # Dibujar el rectángulo en el frame (Color Verde)
    cv2.rectangle(frame, (x_inicio, y_inicio), (x_fin, y_fin), (0, 255, 0), 2)

    # 4. Recortar la imagen (esto sigue la recomendación de tu profesor)
    # Solo mandamos a la IA lo que está dentro del cuadro, ignorando el fondo de tu cuarto
    recorte = frame[y_inicio:y_fin, x_inicio:x_fin]

    # Preprocesar el recorte para la IA
    recorte_redimensionado = cv2.resize(recorte, (96, 96))
    # Keras espera un lote (batch) de imágenes, así que agregamos una dimensión extra: (1, 96, 96, 3)
    img_array = np.expand_dims(recorte_redimensionado, axis=0) 

    # 5. Hacer la predicción
    predicciones = modelo.predict(img_array, verbose=0)
    indice_clase = np.argmax(predicciones[0])
    probabilidad = np.max(predicciones[0]) * 100
    animal_detectado = CLASES[indice_clase]

    # Solo mostrar si la IA está algo segura (ej. > 60%)
    if probabilidad > 60:
        texto = f"{animal_detectado}: {probabilidad:.2f}%"
        # Poner el texto arriba del cuadro
        cv2.putText(frame, texto, (x_inicio, y_inicio - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Mostrar la cámara con todo puesto
    cv2.imshow('Deteccion de Animales', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()