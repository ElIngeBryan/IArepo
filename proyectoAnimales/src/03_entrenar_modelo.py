import tensorflow as tf
from tensorflow.keras import layers, models
import os

# =========================================================
# 1. CONFIGURACIÓN GENERAL
# =========================================================
BATCH_SIZE = 16       # Cantidad de imágenes que procesa la RAM a la vez
IMG_SIZE = (96, 96)   # Resolución optimizada para no saturar la tarjeta gráfica
EPOCHS = 25           # Límite máximo de repasos (se detendrá antes gracias al freno)
DATASET_DIR = "../dataset"

# =========================================================
# 2. CARGA DE DATOS Y OPTIMIZACIÓN (Pipeline)
# =========================================================
print("Cargando imágenes de entrenamiento...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("Cargando imágenes de validación (El examen de la IA)...")
val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

clases = train_dataset.class_names
print(f"\nClases detectadas para entrenar: {clases}")

# Cache y Prefetch: Guarda las imágenes en memoria después de leerlas la primera vez 
# para que las siguientes épocas sean ultrarrápidas y no haya cuellos de botella.
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# =========================================================
# 3. AUMENTO DE DATOS (Data Augmentation)
# =========================================================
# Escudo contra el sobreajuste (overfitting). Modifica las imágenes 
# "al vuelo" para obligar a la IA a aprender formas y no memorizar fotos.
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)), # Efecto espejo
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),              # Desplazamiento
    layers.RandomRotation(0.1),                                                 # Rotación leve
    layers.RandomZoom(0.1),                                                     # Zoom aleatorio
    layers.RandomContrast(0.3),                                                 # Cambios de contraste
    layers.RandomBrightness(0.3),                                               # Cambios de iluminación
])

# =========================================================
# 4. CONSTRUCCIÓN DE LA CNN (El Cerebro)
# =========================================================
modelo = models.Sequential([
    # Inyectamos las modificaciones dinámicas
    data_augmentation,
    
    # Normalización: Pasa los colores de escala 0-255 a 0.0-1.0
    layers.Rescaling(1./255),
    
    # Ruido Gaussiano: Simula estática de cámara web para hacer el modelo más robusto
    layers.GaussianNoise(0.1), 
    
    # --- FASE DE EXTRACCIÓN VISUAL (Filtros) ---
    
    # Capa 1: Detecta líneas y bordes simples
    layers.Conv2D(32, (3, 3), padding='same'),
    layers.LeakyReLU(alpha=0.1), # Mantiene un flujo constante de aprendizaje sin "neuronas muertas"
    layers.MaxPooling2D((2, 2)), # Reduce la imagen a la mitad para ahorrar memoria
    
    # Capa 2: Detecta texturas y combinaciones de bordes
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.LeakyReLU(alpha=0.1),
    layers.MaxPooling2D((2, 2)),
    
    # Capa 3: Detecta formas complejas (patas, ojos, picos)
    layers.Conv2D(128, (3, 3), padding='same'),
    layers.LeakyReLU(alpha=0.1),
    layers.MaxPooling2D((2, 2)),
    
    # --- FASE DE DECISIÓN (Razonamiento) ---
    
    # Aplana los mapas de características a un vector de 1D
    layers.Flatten(),
    
    # Neuronas ocultas que analizan la información extraída
    layers.Dense(128),
    layers.LeakyReLU(alpha=0.1),
    
    # Dropout: Apaga el 50% de las neuronas al azar forzando a todas a aprender por igual
    layers.Dropout(0.5), 
    
    # Capa de Salida: 5 neuronas (una por clase). Softmax da el porcentaje de probabilidad.
    layers.Dense(len(clases), activation='softmax')
])

# =========================================================
# 5. COMPILACIÓN Y ENTRENAMIENTO
# =========================================================
# El optimizador Adam ajusta la velocidad de aprendizaje matemáticamente para converger más rápido
modelo.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

# Imprime la arquitectura en consola
modelo.summary()

# --- SISTEMAS DE SEGURIDAD PARA EL ENTRENAMIENTO ---

# 1. Early Stopping: Detiene el entrenamiento si pasan 4 épocas sin mejorar, ahorrando tiempo.
freno_automatico = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=4, 
    restore_best_weights=True
)

# 2. Model Checkpoint: Guarda automáticamente el modelo SOLO si rompe su récord de precisión.
if not os.path.exists("../modelos"):
    os.makedirs("../modelos")

guardado_seguro = tf.keras.callbacks.ModelCheckpoint(
    filepath="../modelos/modelo_animales.keras",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

print("\nIniciando el entrenamiento del modelo...")
history = modelo.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    # ¡Aquí inyectamos las herramientas de seguridad!
    #callbacks=[freno_automatico, guardado_seguro] 
)

print("\n¡Entrenamiento completado exitosamente y modelo guardado de forma segura!")