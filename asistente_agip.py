# asistente_agip.py
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clase para embeddings personalizados usando scikit-learn
class SimpleEmbeddings:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=768)
        self.fitted = False

    def embed_documents(self, texts):
        if not self.fitted:
            self.tfidf.fit(texts)
            self.fitted = True
        return self.tfidf.transform(texts).toarray().astype(np.float32)

    def embed_query(self, text):
        if not self.fitted:
            self.tfidf.fit([text])
            self.fitted = True
        return self.tfidf.transform([text]).toarray().astype(np.float32)[0]

class AsistenteAGIP:
    """Asistente para consultas sobre trámites y exenciones de AGIP utilizando Claude"""

    def __init__(self, claude_api_key=None, knowledge_base_dir="faiss_index"):
        """
        Inicializa el asistente con Claude y la base de conocimiento
        """
        # Verificar clave API
        api_key = claude_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("No se ha proporcionado una clave API de Anthropic. Por favor, configura la variable de entorno ANTHROPIC_API_KEY o pasa la clave como parámetro.")

        # Inicializar Claude
        self.model = ChatAnthropic(
            model="claude-3-7-sonnet-20250219",
            temperature=0.1,
            anthropic_api_key=api_key,
            max_tokens=1000
        )

        # Inicializar embeddings simples
        self.embeddings = SimpleEmbeddings()

        # Cargar base de conocimiento
        if os.path.exists(knowledge_base_dir):
            self.vector_store = FAISS.load_local(
                folder_path=knowledge_base_dir,
                embeddings=self.embeddings,
                # Añadir este parámetro:
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 5}
            )
            logger.info(f"Base de conocimiento cargada desde {knowledge_base_dir}")
        else:
            raise ValueError(f"No se encontró la base de conocimiento en {knowledge_base_dir}")

        # Plantilla de prompt para consultas
        self.prompt = ChatPromptTemplate.from_template(
            """
            Eres un asistente virtual especializado en trámites y exenciones por discapacidad de AGIP (Administración Gubernamental de Ingresos Públicos).
            
            Tu objetivo es proporcionar información clara, precisa y empática sobre trámites y beneficios fiscales para personas con discapacidad.
            
            Instrucciones:
            - Responde de manera clara y sencilla, evitando jerga técnica innecesaria
            - Muestra empatía hacia las personas con discapacidad y sus familias
            - Si la información específica no está en el contexto, indica claramente que el usuario debería consultar directamente con AGIP
            - Incluye información sobre dónde y cómo realizar los trámites cuando esté disponible
            - Menciona siempre los requisitos documentales necesarios
            - Estructura tus respuestas en párrafos breves y claros
            
            Contexto de la información:
            {context}
            
            Pregunta:
            {question}
            
            Respuesta:
            """
        )

        # Historial de interacciones
        self.history = []

    def answer_question(self, question, k=5, score_threshold=0.2):
        """
        Responde a una pregunta usando RAG con la base de conocimiento
        """
        # Configurar retriever si se modifican los parámetros
        if k != 5:
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": k}
            )

        # Recuperar documentos relevantes
        try:
            relevant_docs = self.retriever.get_relevant_documents(question)
            logger.info(f"Recuperados {len(relevant_docs)} documentos relevantes")

            if not relevant_docs:
                response = "No encontré información específica sobre ese tema en mi base de conocimiento. Te recomiendo consultar directamente en la página oficial de AGIP: https://www.agip.gob.ar/ o llamar al centro de atención telefónica 0800-999-2447."
                self.history.append((question, response))
                return response

            # Crear contexto combinado
            context = "\n\n---\n\n".join([
                f"[Documento: {doc.metadata.get('source', 'Desconocido')}, "
                f"Página: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
                for doc in relevant_docs
            ])

            # Preparar datos para el prompt
            formatted_input = {
                "context": context,
                "question": question
            }

            # Ejecutar el chain
            chain = (
                    RunnablePassthrough()
                    | self.prompt
                    | self.model
                    | StrOutputParser()
            )

            response = chain.invoke(formatted_input)

            # Guardar en historial
            self.history.append((question, response))

            return response

        except Exception as e:
            logger.error(f"Error al responder: {e}")
            return f"Lo siento, ocurrió un error al procesar tu consulta. Por favor, intenta nuevamente o contacta directamente con AGIP."

    def get_history(self):
        """Devuelve el historial de conversación"""
        return self.history