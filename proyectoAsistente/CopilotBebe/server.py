from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle
import re

app = FastAPI(title="Bryan AI - Autocompletado Estructural")

model = tf.keras.models.load_model('modelo_bryan.h5')
with open('vocabulario.pkl', 'rb') as f:
    vocab = pickle.load(f)

token2idx = vocab['token2idx']
idx2token = vocab['idx2token']

class CodePayload(BaseModel):
    context: str

@app.post("/predict")
def predict_next_chunk(data: CodePayload):
    texto = data.context
    
    # 1. INTENTO DE AUTOCOMPLETADO ESTRUCTURAL (Bloques completos)
    # Expresión regular para analizar lo que el usuario está escribiendo AHORA MISMO.
    patron_actual = re.compile(r'(const\s+char\s*\*|int|float)\s+([a-zA-Z_]\w*)\s*(\(.*?\))?\s*$', re.DOTALL)
    match = patron_actual.search(texto)
    
    if match:
        tipo = match.group(1) or ""
        nombre = match.group(2) or ""
        params = match.group(3) or ""
        
        # Lógica de Autocompletado Estructural
        if tipo and nombre and not params:
            secuencia = ['<PAD>', tipo, nombre]
        elif tipo and nombre and params:
            secuencia = [tipo, nombre, params]
        else:
            secuencia = []
            
        if secuencia:
            es_valido = all(t == '<PAD>' or t in token2idx for t in secuencia)
            if es_valido:
                X_input = np.array([[token2idx[t] for t in secuencia]])
                pred = model.predict(X_input, verbose=0)[0]
                
                idx_predicho = np.argmax(pred)
                confianza = pred[idx_predicho]
                
                if confianza > 0.40:
                    sugerencia = idx2token[idx_predicho]
                    
                    if sugerencia.startswith("{"):
                        return {"suggestion": " " + sugerencia}
                    else:
                        return {"suggestion": " " + sugerencia + " "}
                        
    # 2. INTENTO DE AUTOCOMPLETADO PARCIAL (Letra por letra basado en probabilidad)
    # Si el usuario está a la mitad de escribir el nombre (ej. "int su")
    patron_parcial = re.compile(r'(const\s+char\s*\*|int|float)\s+([a-zA-Z_]\w*)$', re.DOTALL)
    match_parcial = patron_parcial.search(texto)
    
    if match_parcial:
        tipo = match_parcial.group(1)
        fragmento = match_parcial.group(2)
        
        if fragmento not in token2idx and tipo in token2idx:
            # El contexto de la IA es el tipo que ya escribió
            secuencia = ['<PAD>', '<PAD>', tipo]
            X_input = np.array([[token2idx[t] for t in secuencia]])
            pred = model.predict(X_input, verbose=0)[0]
            
            # Buscamos la palabra más probable que coincida con las letras que lleva
            mejores_indices = np.argsort(pred)[::-1] 
            for idx in mejores_indices:
                candidato = idx2token[idx]
                if candidato.startswith(fragmento) and candidato != fragmento:
                    # Retornamos solo el resto de la palabra
                    resto = candidato[len(fragmento):]
                    return {"suggestion": resto}

    # 3. AUTOCOMPLETADO BÁSICO DE TIPO (ej. "fl" -> sugiere "oat")
    match_tipo = re.search(r'(^|\n)\s*([a-zA-Z_]\w*)$', texto)
    if match_tipo:
        fragmento_tipo = match_tipo.group(2)
        tipos_conocidos = ["int", "float", "const char*"]
        for t in tipos_conocidos:
            if t.startswith(fragmento_tipo) and t != fragmento_tipo:
                return {"suggestion": t[len(fragmento_tipo):] + " "}
                
    return {"suggestion": ""}

if __name__ == "__main__":
    import uvicorn
    print("Iniciando servidor de Bryan AI en http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)