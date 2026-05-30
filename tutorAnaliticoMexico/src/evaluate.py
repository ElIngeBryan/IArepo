import time
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "../data/vector_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

preguntas = [
    "¿Cuáles son las tres entidades federativas con mayor índice de homicidios dolosos según el corpus?",
    "¿Cómo ha evolucionado la tasa de extorsión y qué sectores son los más afectados?",
    "¿Qué impacto tiene la violencia documentada sobre la tasa de deserción escolar?"
]

def evaluar_tutor(pregunta):
    print(f"\n--- EVALUANDO: {pregunta} ---")
    start_time = time.time()
    
    # Recuperación (Top 3)
    resultados = db.similarity_search(pregunta, k=3)
    contexto = ""
    print("Chunks recuperados:")
    for res in resultados:
        fuente = res.metadata.get('source', 'Desconocido')
        pagina = res.metadata.get('page', 'N/A')
        print(f"- [Fuente: {fuente}, Pág: {pagina}]")
        contexto += f"{res.page_content}\n\n"

    prompt = f"""Usa la siguiente información del corpus para responder a la pregunta. 
    Aplica tus reglas de comportamiento: cita fuentes, sé neutral, usa el método socrático si aplica, 
    y si la respuesta no está en el contexto, di exactamente: 'La información proporcionada en el corpus no detalla este aspecto'.
    
    Contexto:
    {contexto}
    
    Pregunta: {pregunta}
    """

    # Inferencia al modelo local
    response = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system", "content": "Eres un Tutor Analítico experto en seguridad pública."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1 
    )

    latencia = round(time.time() - start_time, 2)
    print(f"\nRespuesta del Tutor:\n{response.choices[0].message.content}")
    print(f"Latencia: {latencia} segundos")
    print("-" * 50)

for p in preguntas:
    evaluar_tutor(p)