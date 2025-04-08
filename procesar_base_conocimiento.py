# procesar_base_conocimiento.py
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings  # Importar la clase base
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clase para embeddings personalizados implementando la interfaz correcta
class SimpleEmbeddings(Embeddings):
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=768)
        self.fitted = False

    def embed_documents(self, texts):
        """Convierte documentos a vectores de embeddings"""
        if not self.fitted:
            self.tfidf.fit(texts)
            self.fitted = True
        return self.tfidf.transform(texts).toarray().astype(np.float32).tolist()

    def embed_query(self, text):
        """Convierte una consulta a un vector de embedding"""
        if not self.fitted:
            self.tfidf.fit([text])
            self.fitted = True
        return self.tfidf.transform([text]).toarray().astype(np.float32)[0].tolist()

class ProcesadorPDFs:
    def __init__(self):
        """Inicializa el procesador de PDFs con embeddings simples"""
        # Configurar embeddings
        self.embeddings = SimpleEmbeddings()

        # Configurar divisor de texto
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=128,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=True
        )

    def procesar_directorio(self, directorio_pdfs, directorio_salida="faiss_index"):
        """Procesa todos los PDFs en un directorio"""
        logger.info(f"Procesando PDFs en {directorio_pdfs}")

        all_docs = []

        # Procesar cada PDF en el directorio
        for filename in os.listdir(directorio_pdfs):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(directorio_pdfs, filename)
                logger.info(f"Procesando: {filename}")

                try:
                    # Cargar el PDF
                    docs = PyPDFLoader(file_path=file_path).load()

                    # Agregar metadatos enriquecidos
                    for i, doc in enumerate(docs):
                        doc.metadata["doc_id"] = f"{filename}_{i}"
                        doc.metadata["source"] = filename
                        doc.metadata["page"] = doc.metadata.get("page", i)

                    all_docs.extend(docs)
                    logger.info(f"PDF {filename} procesado correctamente ({len(docs)} páginas)")

                except Exception as e:
                    logger.error(f"Error procesando {filename}: {e}")

        if not all_docs:
            logger.warning("No se encontraron documentos para procesar")
            return None

        # Dividir documentos en chunks
        chunks = self.text_splitter.split_documents(all_docs)
        logger.info(f"Se crearon {len(chunks)} fragmentos de texto")

        # Crear vector store con FAISS
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        # Guardar el índice FAISS
        if not os.path.exists(directorio_salida):
            os.makedirs(directorio_salida)

        vector_store.save_local(directorio_salida)

        logger.info(f"Base de conocimiento creada exitosamente en {directorio_salida}")
        return vector_store

# Ejecutar el procesamiento si se ejecuta el script directamente
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Procesa PDFs para crear una base de conocimiento vectorial")
    parser.add_argument("--dir", required=True, help="Directorio donde se encuentran los PDFs")
    parser.add_argument("--output", default="faiss_index", help="Directorio donde guardar la base de conocimiento")

    args = parser.parse_args()

    procesador = ProcesadorPDFs()
    procesador.procesar_directorio(args.dir, args.output)