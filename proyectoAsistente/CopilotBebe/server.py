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
    # Extrae el texto completo del archivo actual enviado por la extensión de VS Code
    texto = data.context
    
    # =========================================================================
    # INTENTO 1: AUTOCOMPLETADO ESTRUCTURAL (Predicción de bloques completos)
    # =========================================================================
    
    # Expresión regular que busca si el usuario ya escribió una estructura válida en la línea actual.
    # Captura: Grupo 1 (Tipo de dato), Grupo 2 (Nombre) y Grupo 3 (Parámetros si ya existen).
    patron_actual = re.compile(r'(const\s+char\s*\*|int|float)\s+([a-zA-Z_]\w*)\s*(\(.*?\))?\s*$', re.DOTALL)
    match = patron_actual.search(texto)
    
    if match:
        # Extrae de forma limpia los componentes de la función detectada
        tipo = match.group(1) or ""
        nombre = match.group(2) or ""
        params = match.group(3) or ""
        
        # PIPELINE DE DATOS: Construye la secuencia temporal exacta de 3 pasos (SEQ_LENGTH = 3)
        if tipo and nombre and not params:
            # Caso A: El usuario escribió "int sumar" -> Contexto lógico para predecir Parámetros
            secuencia = ['<PAD>', tipo, nombre]
        elif tipo and nombre and params:
            # Caso B: El usuario escribió "int sumar(int a, int b)" -> Contexto lógico para predecir el Cuerpo {}
            secuencia = [tipo, nombre, params]
        else:
            secuencia = []
            
        if secuencia:
            # FILTRO DE SEGURIDAD: Verifica que todas las palabras de la secuencia existan en el vocabulario entrenado
            es_valido = all(t == '<PAD>' or t in token2idx for t in secuencia)
            if es_valido:
                # VECTORIZACIÓN: Convierte las palabras de texto en sus índices numéricos y crea un Tensor (matriz Numpy)
                X_input = np.array([[token2idx[t] for t in secuencia]])

                # INFERENCIA DE LA IA: La matriz numérica entra a la capa Embedding y luego es procesada 
                # por la capa SimpleRNN usando su memoria secuencial. Devuelve un vector con las probabilidades de todo el vocabulario.
                pred = model.predict(X_input, verbose=0)[0]
                
                # POST-PROCESAMIENTO: Extrae el índice numérico que obtuvo el porcentaje de probabilidad más alto
                idx_predicho = np.argmax(pred)
                confianza = pred[idx_predicho] # Guarda el nivel de certeza matemática (de 0.0 a 1.0)
                
                # UMBRAL DE CERTEZA: Solo si la Vanilla RNN está más de un 40% segura, se aprueba la sugerencia
                if confianza > 0.40:
                    # Traduce el índice numérico de vuelta a texto plano (ej: 3 -> "{ return a + b; }")
                    sugerencia = idx2token[idx_predicho]
                    
                    # Formatea estéticamente los espacios dependiendo de si es un bloque de llaves o parámetros
                    if sugerencia.startswith("{"):
                        return {"suggestion": " " + sugerencia}
                    else:
                        return {"suggestion": " " + sugerencia + " "}
                        
    # =========================================================================
    # INTENTO 2: AUTOCOMPLETADO PARCIAL (Letra por letra asistido por la RNN)
    # =========================================================================
    
    # Se activa si el usuario está a mitad de escribir el nombre de la función (Ej: "int su")
    # Exige que el tipo de dato ya esté completo y seguido por caracteres de una palabra incompleta.
    patron_parcial = re.compile(r'(const\s+char\s*\*|int|float)\s+([a-zA-Z_]\w*)$', re.DOTALL)
    match_parcial = patron_parcial.search(texto)
    
    if match_parcial:
        tipo = match_parcial.group(1)       # Ejemplo: "int"
        fragmento = match_parcial.group(2)  # Ejemplo: "su"
        
        # Valida que el fragmento aún no sea una palabra completa del vocabulario y que el tipo sí exista
        if fragmento not in token2idx and tipo in token2idx:
            # Crea un contexto simulado basándose únicamente en el tipo de dato que el usuario declaró
            secuencia = ['<PAD>', '<PAD>', tipo]
            X_input = np.array([[token2idx[t] for t in secuencia]])
            
            # Consulta a la RNN qué nombres de funciones suelen declararse después de ese tipo de dato
            pred = model.predict(X_input, verbose=0)[0]
            
            # ORDENAMIENTO DE PROBABILIDADES: Ordena los índices del vocabulario de mayor a menor probabilidad
            mejores_indices = np.argsort(pred)[::-1] 
            
            # Recorre las opciones más probables calculadas por la IA para encontrar la adecuada
            for idx in mejores_indices:
                candidato = idx2token[idx]
                # Si el candidato matemático empieza con las letras que lleva escritas el usuario (ej: "sumar" empieza con "su")
                if candidato.startswith(fragmento) and candidato != fragmento:
                    # FILTRA EL RESTO: Resta lo que ya escribió el usuario para devolver únicamente lo que falta
                    # Ejemplo: Si el candidato es "sumar" y escribió "su", calcula "sumar"[2:] -> devuelve "mar"
                    resto = candidato[len(fragmento):]
                    return {"suggestion": resto}

    # =========================================================================
    # INTENTO 3: AUTOCOMPLETADO BÁSICO DE TIPO (Filtro puramente algorítmico)
    # =========================================================================
    
    # Se activa al inicio de una línea limpia si el usuario teclea las primeras letras de un tipo (Ej: "fl")
    match_tipo = re.search(r'(^|\n)\s*([a-zA-Z_]\w*)$', texto)
    if match_tipo:
        fragmento_tipo = match_tipo.group(2) # Ejemplo: "fl"
        tipos_conocidos = ["int", "float", "const char*"]
        
        # Compara de forma estricta contra las palabras reservadas del lenguaje C soportadas por el modelo
        for t in tipos_conocidos:
            if t.startswith(fragmento_tipo) and t != fragmento_tipo:
                # Si coincide, autocompleta el resto del tipo de dato más un espacio para continuar fluidamente
                # Ejemplo: De "fl" calcula "float"[2:] -> devuelve "oat "
                return {"suggestion": t[len(fragmento_tipo):] + " "}
                
    # Si ninguna de las 3 capas lógicas encuentra una coincidencia con alta certeza, el servidor guarda silencio
    return {"suggestion": ""}

if __name__ == "__main__":
    import uvicorn
    print("Iniciando servidor de Bryan AI en http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)