import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_PATH = "../data/raw"
CHROMA_PATH = "../data/vector_db"

print("Cargando PDFs...")
loader = PyPDFDirectoryLoader(DATA_PATH)
documents = loader.load()

# Chunking inyectando metadatos de página
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
)
chunks = text_splitter.split_documents(documents)
print(f"Se generaron {len(chunks)} chunks.")

# Embeddings ligeros y rápidos
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Vectorizando y guardando en ChromaDB...")
db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
print("Base de datos vectorial creada con éxito.")