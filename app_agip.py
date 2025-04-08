# app_agip.py
import streamlit as st
import os
import time
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
    
    .demo-mode-banner {
        background-color: #FFF8E1;
        color: #FF8F00;
        padding: 10px 15px;
        border-radius: 5px;
        border-left: 4px solid #FFA000;
        margin-bottom: 20px;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Implementar respuestas de respaldo
def get_fallback_response(question):
    """Proporciona respuestas predefinidas cuando el modo de respaldo está activado"""
    question_lower = question.lower()

    if "exención" in question_lower or "exencion" in question_lower:
        return (
            "Para solicitar una exención por discapacidad en AGIP, necesitarás presentar:\n\n"
            "1. Certificado Único de Discapacidad (CUD) vigente\n"
            "2. DNI del titular y de la persona con discapacidad\n"
            "3. Documentación que acredite la titularidad del bien (vehículo o inmueble)\n\n"
            "Los trámites se pueden realizar en cualquier Centro de Atención AGIP con turno previo."
        )
    elif "documento" in question_lower or "requisito" in question_lower:
        return (
            "Los documentos necesarios para trámites por discapacidad incluyen:\n\n"
            "- Certificado Único de Discapacidad vigente\n"
            "- DNI de la persona con discapacidad y del titular del bien\n"
            "- Título de propiedad o documentación del vehículo según corresponda\n"
            "- Formulario de solicitud de exención completado\n\n"
            "Recuerda que toda la documentación debe estar actualizada y en buen estado."
        )
    elif "donde" in question_lower or "dónde" in question_lower or "lugar" in question_lower:
        return (
            "Los trámites por discapacidad se pueden realizar en:\n\n"
            "- Cualquier Centro de Atención AGIP (con turno previo)\n"
            "- Algunos trámites pueden iniciarse online a través de la web oficial: https://www.agip.gob.ar/\n\n"
            "Para mayor comodidad, te recomendamos sacar turno con anticipación a través del sistema de turnos online."
        )
    elif "impuesto" in question_lower or "tributo" in question_lower:
        return (
            "Las personas con discapacidad pueden solicitar exenciones en los siguientes impuestos:\n\n"
            "- Impuesto automotor (para vehículos adaptados o destinados a traslado)\n"
            "- ABL (Alumbrado, Barrido y Limpieza) para la vivienda única\n"
            "- Patentes, en casos específicos\n\n"
            "Cada impuesto tiene requisitos particulares que deben cumplirse."
        )
    elif "renovar" in question_lower or "renovación" in question_lower:
        return (
            "Para renovar una exención por discapacidad, debes:\n\n"
            "1. Presentar el CUD actualizado (si estaba por vencer)\n"
            "2. Completar el formulario de renovación de exención\n"
            "3. Adjuntar comprobante de domicilio actualizado\n\n"
            "Es importante iniciar el trámite antes del vencimiento de la exención actual."
        )
    elif "plazo" in question_lower or "vencimiento" in question_lower or "fecha" in question_lower:
        return (
            "Los plazos importantes para exenciones por discapacidad son:\n\n"
            "- Las exenciones deben solicitarse dentro del año fiscal en curso\n"
            "- La renovación debe realizarse antes del vencimiento del beneficio\n"
            "- El CUD debe estar vigente durante todo el período de la exención\n\n"
            "Te recomendamos iniciar los trámites con al menos 30 días de anticipación."
        )
    else:
        return (
            "En esta versión de demostración, puedo responder preguntas básicas sobre exenciones, "
            "documentos requeridos, lugares de trámite e impuestos que pueden ser eximidos para "
            "personas con discapacidad. Para información más específica, por favor consulta "
            "directamente con AGIP en su sitio oficial: https://www.agip.gob.ar/ o llamando al 0800-999-2447."
        )

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
                if st.session_state.get("fallback_mode", False):
                    # Modo de respaldo ya activado
                    time.sleep(1)  # Simular procesamiento
                    response = get_fallback_response(user_text)
                else:
                    # Intentar usar el asistente real
                    try:
                        response = st.session_state["assistant"].answer_question(
                            user_text,
                            k=st.session_state.get("retrieval_k", 5)
                        )
                    except Exception as e:
                        # Si falla, activar modo de respaldo
                        st.warning(f"Error en la búsqueda. Activando modo de respaldo.")
                        st.session_state["fallback_mode"] = True
                        response = get_fallback_response(user_text)
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
        st.session_state["fallback_mode"] = False  # Añadir modo de respaldo

        # Inicializar el asistente
        with st.spinner("Iniciando el asistente..."):
            api_key = os.environ.get("ANTHROPIC_API_KEY")

            if not api_key:
                st.warning("⚠️ No se ha configurado la clave API de Anthropic. Funcionando en modo de respaldo con respuestas predefinidas.")
                st.session_state["fallback_mode"] = True
                st.session_state["messages"].append((
                    "¡Hola! Soy el asistente virtual de AGIP (versión de demostración). "
                    "Puedo responder preguntas básicas sobre trámites y exenciones por discapacidad. "
                    "¿En qué puedo ayudarte hoy?",
                    False, "neutral"
                ))
            else:
                try:
                    st.session_state["assistant"] = AsistenteAGIP(claude_api_key=api_key)
                    st.session_state["messages"].append((
                        "¡Hola! Soy el asistente virtual de AGIP especializado en trámites y exenciones por discapacidad. "
                        "Puedo ayudarte a entender los requisitos, procedimientos y beneficios disponibles. "
                        "¿En qué puedo ayudarte hoy?",
                        False, "neutral"
                    ))
                except Exception as e:
                    st.error(f"Error al iniciar el asistente: {str(e)}")
                    st.warning("Funcionando en modo de respaldo con respuestas predefinidas.")
                    st.session_state["fallback_mode"] = True
                    st.session_state["messages"].append((
                        "¡Hola! Soy el asistente virtual de AGIP (versión de demostración). "
                        "Puedo responder preguntas básicas sobre trámites y exenciones por discapacidad. "
                        "¿En qué puedo ayudarte hoy?",
                        False, "neutral"
                    ))

    # Mostrar banner de modo de respaldo si está activo
    if st.session_state.get("fallback_mode", False):
        st.markdown("""
        <div class="demo-mode-banner">
            <strong>Modo demostración activo</strong>: Funcionando con respuestas predefinidas. Las funciones de RAG y Claude API no están disponibles en este modo.
        </div>
        """, unsafe_allow_html=True)

    # Sidebar con configuración
    with st.sidebar:
        st.header("Configuración")

        # En el modo de respaldo, permitir cambiar manualmente al modo normal
        if st.session_state.get("fallback_mode", False):
            if st.button("Intentar usar API de Claude"):
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if api_key:
                    try:
                        st.session_state["assistant"] = AsistenteAGIP(claude_api_key=api_key)
                        st.session_state["fallback_mode"] = False
                        st.success("¡Conectado a Claude API exitosamente!")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error al conectar con Claude API: {str(e)}")
                else:
                    st.error("No se ha configurado la clave API de Anthropic.")

        # Control de número de documentos solo visible en modo normal
        if not st.session_state.get("fallback_mode", False):
            st.session_state["retrieval_k"] = st.slider(
                "Número de documentos a consultar",
                min_value=1,
                max_value=10,
                value=5
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
        #if len(st.session_state["messages"]) <= 1:
            #st.markdown("### Algunas preguntas que puedes hacer:")
            #col1, col2 = st.columns(2)

            #with col1:
                #if st.button("¿Qué documentos necesito para solicitar la exención por discapacidad?"):
                    #st.session_state["user_input"] = "¿Qué documentos necesito para solicitar la exención por discapacidad?"
                    #process_input()

                #if st.button("¿Dónde puedo realizar los trámites por discapacidad?"):
                    #st.session_state["user_input"] = "¿Dónde puedo realizar los trámites por discapacidad?"
                    #process_input()

            #with col2:
                #if st.button("¿Qué impuestos pueden ser eximidos por discapacidad?"):
                    #st.session_state["user_input"] = "¿Qué impuestos pueden ser eximidos por discapacidad?"
                    #process_input()

                #if st.button("¿Cuál es el proceso para renovar una exención?"):
                    #st.session_state["user_input"] = "¿Cuál es el proceso para renovar una exención?"
                    #process_input()

if __name__ == "__main__":
    main()