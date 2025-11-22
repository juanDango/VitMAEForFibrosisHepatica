import streamlit as st
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor
from pathlib import Path


# -----------------------------
# Paths robustos
# -----------------------------
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "modelos"

MODEL_3_PATH = MODELS_DIR / "3_clases.onnx"
MODEL_5_PATH = MODELS_DIR / "5_clases.onnx"
PREPROCESSOR_PATH = MODELS_DIR  # carpeta con preprocessor_config.json


# -----------------------------
# Config general
# -----------------------------
st.set_page_config(
    page_title="Fibrosis Hepática | Clasificador por Ecografía",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CSS suave (mini front)
# -----------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.hero {
    background: linear-gradient(90deg, #0ea5e9 0%, #22c55e 100%);
    padding: 1.2rem 1.5rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1rem;
}
.hero h1 {font-size: 2rem; margin: 0;}
.hero p {font-size: 1rem; opacity: 0.95; margin-top: 0.3rem;}
.card {
    background: #ffffff;
    border: 1px solid #e6e6e6;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.small {font-size: 0.9rem; color:#555;}
.badge {
    display:inline-block; padding:0.2rem 0.6rem; border-radius:999px;
    font-size:0.85rem; font-weight:600;
}
.badge-high {background:#dcfce7; color:#166534;}
.badge-med {background:#fef9c3; color:#854d0e;}
.badge-low {background:#fee2e2; color:#991b1b;}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Carga cacheada de recursos
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_preprocessor():
    return AutoImageProcessor.from_pretrained(str(PREPROCESSOR_PATH))

@st.cache_resource(show_spinner=False)
def load_sessions():
    sess3 = ort.InferenceSession(str(MODEL_3_PATH), providers=["CPUExecutionProvider"])
    sess5 = ort.InferenceSession(str(MODEL_5_PATH), providers=["CPUExecutionProvider"])
    return sess3, sess5

image_processor = load_preprocessor()
session_3, session_5 = load_sessions()


# -----------------------------
# Funciones auxiliares
# -----------------------------
def preprocess(image: Image.Image):
    """Aplica el mismo preprocesamiento usado en entrenamiento."""
    encoded = image_processor(image, return_tensors="np")
    return encoded["pixel_values"]  # [1, 3, H, W]

def predict(session: ort.InferenceSession, image: Image.Image):
    """
    Ejecuta inferencia sobre una imagen usando una sesión ONNX.

    Si el modelo exportó 'probabilities', se usan directamente.
    Si solo exportó 'logits', se aplica softmax manualmente.
    """
    inputs = preprocess(image)

    output_metadata = session.get_outputs()
    output_names = [o.name for o in output_metadata]
    outputs = session.run(output_names, {"pixel_values": inputs})

    if "probabilities" in output_names:
        idx = output_names.index("probabilities")
        probs = outputs[idx][0]
    elif "logits" in output_names:
        idx = output_names.index("logits")
        logits = outputs[idx][0]
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
    else:
        raise RuntimeError(
            f"El modelo ONNX no tiene ni 'probabilities' ni 'logits' como salida. "
            f"Salidas disponibles: {output_names}"
        )

    return probs  # vector 1D de tamaño num_clases

def plot_probabilities(probabilities, labels, title):
    """Barplot horizontal para mejor lectura clínica."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(labels, probabilities)
    ax.set_xlim([0, 1])
    ax.set_xlabel("Probabilidad")
    ax.set_title(title)
    for i, v in enumerate(probabilities):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center")
    plt.tight_layout()
    return fig

def confidence_badge(p):
    """Clasifica la confianza para semáforo UX."""
    if p >= 0.75:
        return "Alta", "badge badge-high"
    if p >= 0.50:
        return "Media", "badge badge-med"
    return "Baja", "badge badge-low"


# -----------------------------
# HERO / Encabezado
# -----------------------------
st.markdown("""
<div class="hero">
  <h1>Clasificador de Fibrosis Hepática por Ecografía</h1>
  <p>
    Sistema de apoyo clínico basado en Vision Transformers (ViT) preentrenados con 
    ViTMAE (auto-supervisión) para detectar estadios tempranos de fibrosis hepática.
  </p>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Sidebar de configuración
# -----------------------------
st.sidebar.header("Configuración")
mode = st.sidebar.radio(
    "Modelo a utilizar",
    ["Ambos modelos", "Solo 3 clases", "Solo 5 clases"],
    index=0
)

show_probs = st.sidebar.checkbox("Mostrar probabilidades", value=True)
warn_threshold = st.sidebar.slider(
    "Umbral de confianza para alerta",
    0.0, 1.0, 0.60, 0.05
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Este sistema es una herramienta de apoyo clínico y no reemplaza el criterio médico."
)


# -----------------------------
# Acerca del proyecto
# -----------------------------
with st.expander("Acerca del proyecto"):
    st.write("""
**Objetivo**  
Desarrollar un modelo de Deep Learning capaz de identificar estadios tempranos de fibrosis hepática en imágenes de ecografía.

**Enfoque técnico**  
El sistema utiliza **Vision Transformers (ViT)** con preentrenamiento auto-supervisado mediante **ViTMAE (Masked Autoencoder)**, lo que permite obtener mejores representaciones incluso con conjuntos de datos pequeños. Sobre el encoder ajustado se añade una cabeza clasificadora entrenada por *transfer learning*.

**Modelos disponibles**
- **5 clases:** F0 (ausencia de fibrosis), F1–F3 (grados progresivos de fibrosis), F4 (cirrosis).  
- **3 clases:** Sano (F0), Fibrosis leve/moderada (F1–F3), Cirrosis (F4).

**Interpretación**  
El modelo produce probabilidades mediante *softmax*, que representan la estimación del estadio más probable a partir de la ecografía.
    """)



# -----------------------------
# Zona de carga de imagen
# -----------------------------
st.subheader("Cargar ecografía")
uploaded_file = st.file_uploader(
    "Sube una imagen (.jpg, .png)",
    type=["jpg", "jpeg", "png"]
)

if not uploaded_file:
    st.info("Por favor carga una ecografía para obtener una predicción.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")

col_img, col_info = st.columns([1.2, 1])

with col_img:
    st.image(image, caption="Ecografía cargada", use_container_width=True)



# -----------------------------
# Inferencia
# -----------------------------
labels_3 = ["Sano", "Fibrosis leve", "Cirrosis"]
labels_5 = ["F0", "F1", "F2", "F3", "F4"]

with st.spinner("Ejecutando modelo(s)..."):
    probs_3 = predict(session_3, image) if mode != "Solo 5 clases" else None
    probs_5 = predict(session_5, image) if mode != "Solo 3 clases" else None


# -----------------------------
# Resultados
# -----------------------------
st.subheader("Resultados")
cols = st.columns(2)

if probs_3 is not None:
    pred_idx_3 = int(np.argmax(probs_3))
    pred_3 = labels_3[pred_idx_3]
    conf_3 = float(probs_3[pred_idx_3])
    conf_txt_3, conf_class_3 = confidence_badge(conf_3)

    with cols[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Modelo 3 clases")
        st.markdown(f"**Predicción:** `{pred_3}`")
        st.markdown(
            f"**Confianza:** {conf_3:.2f} "
            f"<span class='{conf_class_3}'>{conf_txt_3}</span>",
            unsafe_allow_html=True
        )

        if conf_3 < warn_threshold:
            st.warning("Confianza baja. Interpreta este resultado con cautela.")
        else:
            st.success("Confianza adecuada para apoyo diagnóstico.")

        if show_probs:
            st.pyplot(plot_probabilities(probs_3, labels_3, "Probabilidades (3 clases)"))

        st.markdown("</div>", unsafe_allow_html=True)


if probs_5 is not None:
    pred_idx_5 = int(np.argmax(probs_5))
    pred_5 = labels_5[pred_idx_5]
    conf_5 = float(probs_5[pred_idx_5])
    conf_txt_5, conf_class_5 = confidence_badge(conf_5)

    with cols[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Modelo 5 clases")
        st.markdown(f"**Estadio predicho:** `{pred_5}`")
        st.markdown(
            f"**Confianza:** {conf_5:.2f} "
            f"<span class='{conf_class_5}'>{conf_txt_5}</span>",
            unsafe_allow_html=True
        )

        if conf_5 < warn_threshold:
            st.warning("Confianza baja. Considera confirmar con otras pruebas.")
        else:
            st.success("Confianza adecuada para apoyo diagnóstico.")

        if show_probs:
            st.pyplot(plot_probabilities(probs_5, labels_5, "Probabilidades (5 clases)"))

        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Footer clínico
# -----------------------------
st.markdown("---")
st.caption(
    "Esta herramienta es un sistema de apoyo a la decisión clínica. "
    "No reemplaza evaluación médica, historia clínica ni pruebas complementarias."
)
