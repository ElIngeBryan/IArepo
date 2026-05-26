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
    
    # Expresión regular para analizar lo que el usuario está escribiendo AHORA MISMO.
    # Busca hacia atrás si el usuario ya escribió el Tipo y el Nombre.
    # Los parámetros son opcionales (por si apenas va a escribirlos).
    patron_actual = re.compile(r'(const\s+char\s*\*|int|float)\s+([a-zA-Z_]\w*)\s*(\(.*?\))?\s*$', re.DOTALL)
    match = patron_actual.search(texto)
    
    if not match:
        # Si no estamos declarando una función de C conocida, la IA se queda callada.
        return {"suggestion": ""}
        
    tipo = match.group(1) or ""
    nombre = match.group(2) or ""
    params = match.group(3) or ""
    
    # Lógica de Autocompletado Estructural
    if tipo and nombre and not params:
        # Ya escribió "int sumar" -> Queremos sugerir los parámetros "(int a, int b)"
        secuencia = ['<PAD>', tipo, nombre]
    elif tipo and nombre and params:
        # Ya escribió "int sumar(int a, int b)" -> Queremos sugerir el cuerpo "{ return a+b; }"
        secuencia = [tipo, nombre, params]
    else:
        return {"suggestion": ""}
        
    # Validar que lo que escribió el usuario exista en el dataset original
    for t in secuencia:
        if t != '<PAD>' and t not in token2idx:
            # Si escribe un nombre inventado como "int funcion_nueva", guardamos silencio
            return {"suggestion": ""}
            
    X_input = np.array([[token2idx[t] for t in secuencia]])
    pred = model.predict(X_input, verbose=0)[0]
    
    idx_predicho = np.argmax(pred)
    confianza = pred[idx_predicho]
    
    # Como la arquitectura es limpia y exacta, la confianza suele ser >90%
    if confianza > 0.40:
        sugerencia = idx2token[idx_predicho]
        
        # Agregamos un espacio bonito antes de la sugerencia si es el cuerpo de la función
        if sugerencia.startswith("{"):
            return {"suggestion": " " + sugerencia}
        else:
            return {"suggestion": " " + sugerencia + " "}
        
    return {"suggestion": ""}

if __name__ == "__main__":
    import uvicorn
    print("Iniciando servidor de Bryan AI en http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)