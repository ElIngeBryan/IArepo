import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding, Input

print("Cargando dataset...")
with open("dataset.c", "r", encoding="utf-8") as f:
    texto = f.read()

# Expresión regular que divide el código en 4 super-tokens lógicos:
# 1: Tipo (int, float, const char*)
# 2: Nombre de la función
# 3: Parámetros (con todo y paréntesis)
# 4: Cuerpo de la función (con todo y llaves)
# CORRECCIÓN APLICADA: \{.*?^\} y re.MULTILINE para leer funciones con llaves internas
patron = re.compile(r'(const\s+char\s*\*|int|float)\s+([a-zA-Z_]\w*)\s*(\(.*?\))\s*(\{.*?^\})', re.DOTALL | re.MULTILINE)
funciones = patron.findall(texto)

print(f"Se estructuraron {len(funciones)} funciones en super-tokens lógicos.")

# Crear el vocabulario con todos los bloques únicos
vocab_set = set(['<PAD>'])
for f in funciones:
    vocab_set.update(f) # Añade tipos, nombres, params y cuerpos al vocabulario
    
vocab = sorted(list(vocab_set))
token2idx = {t: i for i, t in enumerate(vocab)}
idx2token = {i: t for i, t in enumerate(vocab)}

with open('vocabulario.pkl', 'wb') as f:
    pickle.dump({'token2idx': token2idx, 'idx2token': idx2token}, f)

# La memoria necesaria ahora es minúscula. 
# Solo necesitamos ver 3 super-tokens hacia atrás para predecir el siguiente.
SEQ_LENGTH = 3 
X_list = []
y_list = []

print("Creando red de conceptos lógicos...")
for tipo, nombre, params, cuerpo in funciones:
    # 1er Paso: Dado el Tipo y el Nombre, predecir los Parámetros
    X_list.append(['<PAD>', tipo, nombre])
    y_list.append(params)
    
    # 2do Paso: Dado el Tipo, Nombre y Parámetros, predecir el Cuerpo entero
    X_list.append([tipo, nombre, params])
    y_list.append(cuerpo)

# Forzamos el tipo int32 para que Keras 3 no tenga dudas de la estructura
X = np.array([[token2idx[t] for t in seq] for seq in X_list], dtype=np.int32)
y = tf.keras.utils.to_categorical([token2idx[t] for t in y_list], num_classes=len(vocab))


model = Sequential([
    # Define la entrada: Espera recibir secuencias exactas de 3 pasos lógicos.
    Input(shape=(SEQ_LENGTH,)),
    
    # 1. ALGORITMO DE VECTORIZACIÓN: Convierte los números en vectores matemáticos 
    # de 64 dimensiones para que la IA agrupe conceptos lógicos similares.
    Embedding(input_dim=len(vocab), output_dim=64),
    
    # 2. ALGORITMO RECURRENTE (VANILLA RNN): La memoria a corto plazo (128 neuronas).
    # Se retroalimenta a sí misma para analizar el orden temporal de las palabras.
    SimpleRNN(128),
    
    # 3. ALGORITMO DE PROBABILIDAD (SOFTMAX): Convierte los cálculos crudos de la red 
    # en porcentajes exactos de probabilidad (0 a 100%) para sugerir el código correcto.
    Dense(len(vocab), activation='softmax')
])

# 4. ALGORITMOS DE EVALUACIÓN Y AJUSTE:
# - adam: Optimizador que calcula derivadas para ajustar los "pesos" rápidamente.
# - categorical_crossentropy: Algoritmo que castiga matemáticamente los errores al adivinar.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Entrenando la IA (llegará al 100% muy rápido)...")

# 5. ALGORITMO DE ENTRENAMIENTO (BPTT - Backpropagation Through Time):
# Desenrolla la red en el tiempo y ajusta los errores hacia atrás durante 25 pasadas.
model.fit(X, y, batch_size=4, epochs=25)