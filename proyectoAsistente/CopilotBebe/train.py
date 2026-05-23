import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
import pickle
import re

print("Cargando dataset...")
with open("dataset.c", "r", encoding="utf-8") as f:
    dataset_c = f.read()

def tokenizar(texto):
    # Separa palabras, pero captura símbolos, saltos de línea y espacios individualmente
    return re.findall(r'\w+|[^\w\s]|\n|[ ]|\t', texto)

tokens_dataset = tokenizar(dataset_c)
vocabulario = sorted(list(set(tokens_dataset)))
vocab_size = len(vocabulario)

# AQUÍ ES DONDE SE CREA EL token2idx QUE EL SERVIDOR BUSCA
token2idx = {token: idx for idx, token in enumerate(vocabulario)}
idx2token = {idx: token for idx, token in enumerate(vocabulario)}

with open('vocabulario.pkl', 'wb') as f:
    pickle.dump({'token2idx': token2idx, 'idx2token': idx2token, 'vocabulario': vocabulario}, f)

SEQ_LENGTH = 30
STEP = 1

secuencias = []
proximos_tokens = []

for i in range(0, len(tokens_dataset) - SEQ_LENGTH, STEP):
    secuencias.append(tokens_dataset[i : i + SEQ_LENGTH])
    proximos_tokens.append(tokens_dataset[i + SEQ_LENGTH])

X = np.zeros((len(secuencias), SEQ_LENGTH), dtype=np.int32)
y = np.zeros((len(secuencias), vocab_size), dtype=np.float32)

for i, secuencia in enumerate(secuencias):
    for t, token in enumerate(secuencia):
        X[i, t] = token2idx[token]
    y[i, token2idx[proximos_tokens[i]]] = 1.0

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64),
    SimpleRNN(256, return_sequences=False),
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(f"Entrenando con {vocab_size} tokens únicos (incluyendo espacios y saltos)...")
model.fit(X, y, batch_size=32, epochs=80) 
model.save('modelo_bryan.h5')
print("¡Entrenamiento completado! Archivos modelo_bryan.h5 y vocabulario.pkl actualizados.")