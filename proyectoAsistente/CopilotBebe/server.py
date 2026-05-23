from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle
import re

app = FastAPI(title="Bryan AI - Autocompletado Básico")

print("Cargando modelo y dataset...")
model = tf.keras.models.load_model('modelo_bryan.h5')
with open('vocabulario.pkl', 'rb') as f:
    vocab = pickle.load(f)

token2idx = vocab['token2idx']
idx2token = vocab['idx2token']
SEQ_LENGTH = 30

class CodePayload(BaseModel):
    context: str

def tokenizar(texto):
    return re.findall(r'\w+|[^\w\s]|\n|[ ]|\t', texto)

def preparar_entrada(tokens):
    # Rellenamos con saltos de línea para estabilizar la RNN
    pad_size = max(0, SEQ_LENGTH - len(tokens))
    ctx = ['\n'] * pad_size + tokens
    ctx = ctx[-SEQ_LENGTH:]
    return np.array([[token2idx.get(t, token2idx.get('\n', 0)) for t in ctx]])

@app.post("/predict")
def predict_next_words(data: CodePayload):
    texto_entrada = data.context

    # Analizar si el usuario dejó una palabra a medias (ej: "su" para "sumar")
    match_parcial = re.search(r'([a-zA-Z_]\w*)$', texto_entrada)
    palabra_parcial = match_parcial.group(1) if match_parcial else ""

    texto_limpio = texto_entrada[:match_parcial.start()] if match_parcial else texto_entrada
    tokens_contexto = tokenizar(texto_limpio)
    
    # 1. MODO: COMPLETAR PALABRA ACTUAL
    if palabra_parcial:
        input_eval = preparar_entrada(tokens_contexto)
        predicciones = model.predict(input_eval, verbose=0)[0]
        
        indices_validos = [idx for token, idx in token2idx.items() if token.startswith(palabra_parcial) and token != palabra_parcial]
        
        if indices_validos:
            idx_predicho = max(indices_validos, key=lambda idx: predicciones[idx])
            primer_token = idx2token[idx_predicho]
            # Devuelve solo las letras faltantes
            return {"suggestion": primer_token[len(palabra_parcial):]}
        else:
            return {"suggestion": ""}

    # 2. MODO: PREDECIR LA SIGUIENTE PALABRA (Cumplimiento estricto del proyecto)
    input_eval = preparar_entrada(tokens_contexto)
    predicciones = model.predict(input_eval, verbose=0)[0]
    
    idx_predicho = np.argmax(predicciones)
    siguiente_token = idx2token[idx_predicho]
    
    # Simplemente devolvemos el siguiente token que predijo la red
    return {"suggestion": siguiente_token}

if __name__ == "__main__":
    import uvicorn
    print("Iniciando servidor de Bryan AI en http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)