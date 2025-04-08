# app_agip.py
import streamlit as st
import os
from datetime import datetime
from asistente_agip import AsistenteAGIP

# Configuración de la página
st.set_page_config(
    page_title="AGIP - Asistente Trámites Discapacidad",
    page_icon="♿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
custom_css = """
<style>
    :root {
        --primary: #00897B;
        --primary-light: #4DB6AC;
        --primary-dark: #00695C;
        --secondary: #546E7A;
        --background: #F5F5F5;
        --card-bg: #FFFFFF;
        --text-dark: #263238;
        --text-light: #FFFFFF;
        --border-light: #E0E0E0;
        --success: #4CAF50;
        --warning: #FFC107;
        --error: #F44336;
        --info: #2196F3;
    }
    
    .main {
        background-color: var(--background);
        font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
    }
    
    .stSidebar {
        background-color: var(--card-bg);
    }
    
    h1 {
        color: var(--primary);
    }
    
    .user-message {
        background-color: var(--primary-light);
        color: var(--text-light);
        padding: 15px;
        border-radius: 10px;
        border-top-right-radius: 0;
        margin-bottom: 15px;
        text-align: right;
        max-width: 80%;
        margin-left: auto;
    }
    
    .bot-message {
        background-color: var(--card-bg);
        padding: 15px;
        border-radius: 10px;
        border-top-left-radius: 0;
        margin-bottom: 15px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        max-width: 80%;
    }
    
    .message-header {
        display: flex;
        align-items: center;
        margin-bottom: 8px;
    }
    
    .message-avatar {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        margin-right: 10px;
        background-color: var(--primary);
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--text-light);
        font-size: 12px;
        font-weight: 500;
    }
    
    .message-time {
        font-size: 12px;
        color: var(--secondary);
    }
    
    .quick-action-btn {
        background-color: #E0F2F1;
        border: 1px solid var(--primary-light);
        border-radius: 20px;
        padding: 8px 15px;
        font-size: 14px;
        color: var(--primary);
        margin-right: 10px;
        margin-bottom: 10px;
        cursor: pointer;
    }
    
    .quick-action-btn:hover {
        background-color: var(--primary-light);
        color: var(--text-light);
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

def display_messages():
    """Muestra los mensajes del chat con estilo"""
    for i, (msg, is_user, _) in enumerate(st.session_state["messages"]):
        if is_user:
            # Estilo de mensaje de usuario
            current_time = datetime.now().strftime("%H:%M")
            st.markdown(f"""
            <div class="user-message">
                <div class="message-header" style="justify-content: flex-end;">
                    <div class="message-time">{current_time}</div>
                    <div style="width: 10px;"></div>
                    <div class="message-avatar" style="background-color: var(--primary-dark);">U</div>
                </div>
                <div>{msg}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Estilo de mensaje del bot
            current_time = datetime.now().strftime("%H:%M")
            st.markdown(f"""
            <div class="bot-message">
                <div class="message-header">
                    <div class="message-avatar">IA</div>
                    <div>Asistente AGIP</div>
                    <div style="flex-grow: 1;"></div>
                    <div class="message-time">{current_time}</div>
                </div>
                <div>{msg}</div>
            </div>
            """, unsafe_allow_html=True)

def process_input():
    """Procesa la entrada del usuario"""
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        st.session_state["user_input"] = ""

        # Agregar mensaje del usuario
        st.session_state["messages"].append((user_text, True, "neutral"))

        # Obtener respuesta
        with st.session_state["thinking_spinner"], st.spinner("Procesando..."):
            try:
                response = st.session_state["assistant"].answer_question(
                    user_text,
                    k=st.session_state.get("retrieval_k", 5),
                    score_threshold=st.session_state.get("retrieval_threshold", 0.2)
                )
            except Exception as e:
                response = f"Lo siento, ocurrió un error: {str(e)}"

        # Agregar respuesta del asistente
        st.session_state["messages"].append((response, False, "neutral"))

def main():
    """Función principal de la aplicación"""
    st.title("Asistente AGIP - Trámites y Exenciones por Discapacidad")
    st.markdown("<p style='color: var(--secondary);'>Información sobre trámites, beneficios y exenciones para personas con discapacidad</p>", unsafe_allow_html=True)

    # Inicializar el estado de la sesión
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["user_input"] = ""

        # Inicializar el asistente
        with st.spinner("Iniciando el asistente..."):
            try:
                st.session_state["assistant"] = AsistenteAGIP()
                st.session_state["messages"].append((
                    "¡Hola! Soy el asistente virtual de AGIP especializado en trámites y exenciones por discapacidad. "
                    "Puedo ayudarte a entender los requisitos, procedimientos y beneficios disponibles. "
                    "¿En qué puedo ayudarte hoy?",
                    False, "neutral"
                ))
            except Exception as e:
                st.error(f"Error al iniciar el asistente: {e}")
                st.stop()

    # Sidebar con configuración
    with st.sidebar:
        st.header("Configuración")
        st.session_state["retrieval_k"] = st.slider(
            "Número de documentos a consultar",
            min_value=1,
            max_value=10,
            value=5
        )
        st.session_state["retrieval_threshold"] = st.slider(
            "Umbral de relevancia",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05
        )

        if st.button("Limpiar conversación"):
            st.session_state["messages"] = [st.session_state["messages"][0]]  # Mantener solo el mensaje de bienvenida
            st.experimental_rerun()

    # Contenedor principal
    chat_container = st.container()

    with chat_container:
        # Mostrar mensajes
        display_messages()

        # Spinner para cuando está procesando
        st.session_state["thinking_spinner"] = st.empty()

        # Input del usuario
        st.text_input(
            "Escribe tu consulta aquí...",
            key="user_input",
            on_change=process_input
        )

        # Sugerencias de preguntas (solo si no hay mensajes)
        if len(st.session_state["messages"]) <= 1:
            st.markdown("### Algunas preguntas que puedes hacer:")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("¿Qué documentos necesito para solicitar la exención por discapacidad?"):
                    st.session_state["user_input"] = "¿Qué documentos necesito para solicitar la exención por discapacidad?"
                    process_input()

                if st.button("¿Dónde puedo realizar los trámites por discapacidad?"):
                    st.session_state["user_input"] = "¿Dónde puedo realizar los trámites por discapacidad?"
                    process_input()

            with col2:
                if st.button("¿Qué impuestos pueden ser eximidos por discapacidad?"):
                    st.session_state["user_input"] = "¿Qué impuestos pueden ser eximidos por discapacidad?"
                    process_input()

                if st.button("¿Cuál es el proceso para renovar una exención?"):
                    st.session_state["user_input"] = "¿Cuál es el proceso para renovar una exención?"
                    process_input()

if __name__ == "__main__":
    main()