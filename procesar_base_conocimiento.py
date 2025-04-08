# procesar_base_conocimiento.py
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcesadorPDFs:
    def __init__(self, openai_api_key=None):
        """Inicializa el procesador de PDFs con embeddings de OpenAI"""
        # Configurar embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_api_key or os.environ.get("OPENAI_API_KEY")
        )

        # Configurar divisor de texto
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=128,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=True
        )

    def procesar_directorio(self, directorio_pdfs, directorio_salida="chroma_db_agip_discapacidad"):
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
        chunks = filter_complex_metadata(chunks)
        logger.info(f"Se crearon {len(chunks)} fragmentos de texto")

        # Crear vector store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=directorio_salida
        )
        vector_store.persist()

        logger.info(f"Base de conocimiento creada exitosamente en {directorio_salida}")
        return vector_store

# Ejecutar el procesamiento si se ejecuta el script directamente
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Procesa PDFs para crear una base de conocimiento vectorial")
    parser.add_argument("--dir", required=True, help="Directorio donde se encuentran los PDFs")
    parser.add_argument("--output", default="chroma_db_agip_discapacidad", help="Directorio donde guardar la base de conocimiento")
    parser.add_argument("--api_key", help="OpenAI API Key (opcional, también puede usar la variable de entorno OPENAI_API_KEY)")

    args = parser.parse_args()

    procesador = ProcesadorPDFs(openai_api_key=args.api_key)
    procesador.procesar_directorio(args.dir, args.output)