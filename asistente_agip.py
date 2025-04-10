# asistente_agip.py
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clase para embeddings personalizados implementando la interfaz correcta
class SimpleEmbeddings(Embeddings):
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.tfidf = TfidfVectorizer(max_features=dimension)
        self.fitted = False
        self.default_vector = np.zeros(dimension).astype(np.float32).tolist()

    def _ensure_dimension(self, vector):
        """Asegura que el vector tenga la dimensión correcta"""
        if len(vector) < self.dimension:
            return vector + [0.0] * (self.dimension - len(vector))
        elif len(vector) > self.dimension:
            return vector[:self.dimension]
        return vector

    def embed_documents(self, texts):
        try:
            if not self.fitted:
                self.tfidf.fit(texts)
                self.fitted = True

            vectors = self.tfidf.transform(texts).toarray().astype(np.float32)
            return [self._ensure_dimension(v.tolist()) for v in vectors]
        except Exception as e:
            return [self.default_vector for _ in texts]

    def embed_query(self, text):
        try:
            if not self.fitted:
                self.tfidf.fit([text])
                self.fitted = True

            vector = self.tfidf.transform([text]).toarray()[0].astype(np.float32)
            return self._ensure_dimension(vector.tolist())
        except Exception as e:
            return self.default_vector

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
        try:
            self.model = ChatAnthropic(
                model="claude-3-7-sonnet-20250219",
                temperature=0.1,
                anthropic_api_key=api_key,
                max_tokens=1000
            )
            logger.info("Modelo ChatAnthropic inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar ChatAnthropic: {e}")
            logger.error(traceback.format_exc())
            raise

        # Inicializar embeddings simples
        self.embeddings = SimpleEmbeddings(dimension=768)

        # Cargar base de conocimiento
        try:
            if os.path.exists(knowledge_base_dir):
                self.vector_store = FAISS.load_local(
                    folder_path=knowledge_base_dir,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": 5}
                )
                logger.info(f"Base de conocimiento cargada desde {knowledge_base_dir}")
            else:
                raise ValueError(f"No se encontró la base de conocimiento en {knowledge_base_dir}")
        except Exception as e:
            logger.error(f"Error al cargar la base de conocimiento: {e}")
            logger.error(traceback.format_exc())
            raise

        # Plantilla de prompt para consultas
        self.prompt = ChatPromptTemplate.from_template(
            """
            Eres un asistente virtual especializado en trámites y exenciones de AGIP (Administración Gubernamental de Ingresos Públicos).
            
            Tu objetivo es proporcionar información clara, precisa y empática sobre trámites y beneficios fiscales para las personas.
            
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

    def answer_question(self, question, k=5):
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
            logger.info(f"Buscando documentos relevantes para: {question}")

            # Enfoque directo: si hay errores, fallar rápido
            relevant_docs = []
            try:
                # Crear consulta directamente con la misma clase de embeddings
                query_embedding = self.embeddings.embed_query(question)
                # Usar el método search_by_vector directamente
                docs_and_scores = self.vector_store.similarity_search_with_score_by_vector(
                    query_embedding, k=k
                )
                # Extraer solo los documentos
                relevant_docs = [doc for doc, _ in docs_and_scores]
                logger.info(f"Búsqueda exitosa con similarity_search_with_score_by_vector")
            except Exception as e:
                logger.error(f"Error en similarity_search_with_score_by_vector: {e}")
                logger.error(traceback.format_exc())
                # No reintentamos, dejamos que la excepción se propague al try/except exterior
                raise

            logger.info(f"Recuperados {len(relevant_docs)} documentos relevantes")

            if not relevant_docs:
                response = "No encontré información específica sobre ese tema en mi base de conocimiento. Te recomiendo consultar directamente en la página oficial de AGIP: https://www.agip.gob.ar/ o llamar al centro de atención telefónica 0800-999-2447."
                self.history.append((question, response))
                return response

            # Crear contexto combinado
            logger.info("Construyendo contexto a partir de documentos relevantes")
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
            logger.info("Invocando el modelo Claude para generar respuesta")
            chain = (
                    RunnablePassthrough()
                    | self.prompt
                    | self.model
                    | StrOutputParser()
            )

            try:
                response = chain.invoke(formatted_input)
                logger.info("Respuesta generada correctamente")
            except Exception as e:
                logger.error(f"Error al generar respuesta con Claude: {e}")
                logger.error(traceback.format_exc())
                raise

            # Guardar en historial
            self.history.append((question, response))

            return response

        except Exception as e:
            logger.error(f"Error al responder: {e}")
            logger.error(traceback.format_exc())  # Añadir stack trace completo
            # Enviar mensaje más genérico al usuario
            return f"Lo siento, ocurrió un error al procesar tu consulta. Por favor, intenta nuevamente con otra pregunta o contacta directamente con AGIP al 0800-999-2447."

    def get_history(self):
        """Devuelve el historial de conversación"""
        return self.history