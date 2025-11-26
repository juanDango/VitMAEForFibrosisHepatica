import streamlit as st
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor
from pathlib import Path
from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader



ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "modelos"

MODEL_3_PATH = MODELS_DIR / "3_clases.onnx"
MODEL_5_PATH = MODELS_DIR / "5_clases.onnx"
PREPROCESSOR_PATH = MODELS_DIR



# Config general
st.set_page_config(
    page_title="Clasificador de Fibrosis Hepática por Ecografía",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# CSS suave

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
.badge {
    display:inline-block; padding:0.2rem 0.6rem; border-radius:999px;
    font-size:0.85rem; font-weight:600;
}
.badge-high {background:#dcfce7; color:#166534;}
.badge-med {background:#fef9c3; color:#854d0e;}
.badge-low {background:#fee2e2; color:#991b1b;}
</style>
""", unsafe_allow_html=True)



# Carga cacheada de recursos

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



# Funciones auxiliares

def preprocess(image: Image.Image):
    encoded = image_processor(image, return_tensors="np")
    return encoded["pixel_values"]

def predict(session: ort.InferenceSession, image: Image.Image):
    """
    Ejecuta inferencia sobre una imagen usando una sesión ONNX.

    Si el modelo exportó 'probabilities', se usan directamente.
    Si solo exportó 'logits', se aplica softmax manualmente.
    """
    inputs = preprocess(image)

    # nombres de salida del modelo ONNX
    output_metadata = session.get_outputs()
    output_names = [o.name for o in output_metadata]

    # ejecutar todos los outputs disponibles
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
    if p >= 0.75:
        return "Alta", "badge badge-high"
    if p >= 0.50:
        return "Media", "badge badge-med"
    return "Baja", "badge badge-low"



def generate_pdf_report(rows, title="Reporte Fibrosis Hepática"):
    """
    rows: lista de dicts:
      - name
      - pred3, conf3 (pueden ser None)
      - pred5, conf5 (pueden ser None)
      - image (PIL)
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    margin = 40
    y = height - margin

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, title)
    y -= 20

    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 20

    c.setFont("Helvetica", 9)
    c.drawString(margin, y, "Este reporte es apoyo clínico. No reemplaza evaluación médica.")
    y -= 25

    for i, r in enumerate(rows, start=1):
        if y < 200:
            c.showPage()
            y = height - margin

        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, f"{i}. {r['name']}")
        y -= 14

        c.setFont("Helvetica", 10)

        # Mostrar resultados en líneas separadas para que SIEMPRE se vean ambos
        if r.get("pred3") is not None:
            c.drawString(margin, y, f"Modelo 3 clases: {r['pred3']} (confianza: {r['conf3']:.2f})")
            y -= 12

        if r.get("pred5") is not None:
            c.drawString(margin, y, f"Modelo 5 clases: {r['pred5']} (confianza: {r['conf5']:.2f})")
            y -= 12

        # Imagen
        img_pil = r["image"].copy()
        img_pil.thumbnail((220, 220))
        img_io = BytesIO()
        img_pil.save(img_io, format="PNG")
        img_io.seek(0)

        c.drawImage(
            ImageReader(img_io),
            margin, y - 140,
            width=140, height=140,
            preserveAspectRatio=True,
            mask="auto"
        )
        y -= 155

    c.save()
    buffer.seek(0)
    return buffer



st.markdown("""
<div class="hero">
  <h1>Clasificador de Fibrosis Hepática por Ecografía</h1>
  <p>
    Sistema de apoyo clínico basado en Vision Transformers con preentrenamiento auto-supervisado ViTMAE.
  </p>
</div>
""", unsafe_allow_html=True)



st.sidebar.header("Configuración")

input_mode = st.sidebar.radio(
    "Modo de entrada",
    ["Imagen única", "Múltiples imágenes (carpeta)"],
    index=0
)

model_mode = st.sidebar.radio(
    "Modelo a utilizar",
    ["Ambos modelos", "Solo 3 clases", "Solo 5 clases"],
    index=0
)

show_probs = st.sidebar.checkbox("Mostrar probabilidades", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Herramienta de apoyo clínico, no reemplaza criterio médico.")




with st.expander("Acerca del proyecto"):
    st.write("""
Objetivo: identificar estadios tempranos de fibrosis hepática en ecografía.

Método: Vision Transformers preentrenados con ViTMAE y ajustados por transfer learning.

Modelos:
- 5 clases: F0 (sin fibrosis), F1–F3 (fibrosis progresiva), F4 (cirrosis).
- 3 clases: Sano (F0), Fibrosis (F1–F3), Cirrosis (F4).

La salida son probabilidades softmax por clase.
""")


labels_3 = ["Sano", "Fibrosis leve", "Cirrosis"]
labels_5 = ["F0", "F1", "F2", "F3", "F4"]



# MODO 1: Imagen única
if input_mode == "Imagen única":
    st.subheader("Cargar ecografía (una sola imagen)")
    uploaded_file = st.file_uploader("Sube una imagen (.jpg, .png)", type=["jpg", "jpeg", "png"])

    if not uploaded_file:
        st.info("Carga una ecografía para obtener la predicción.")
        st.stop()

    image = Image.open(uploaded_file).convert("RGB")
    col_img, col_info = st.columns([1.2, 1])

    with col_img:
        st.image(image, caption="Ecografía cargada", use_container_width=True)

    with col_info:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("Recomendaciones:")
        st.markdown("""
        - Imagen centrada en hígado.
        - Evitar bordes y texto.
        - Buena resolución/contraste.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.spinner("Ejecutando modelo(s)..."):
        probs_3 = predict(session_3, image) if model_mode != "Solo 5 clases" else None
        probs_5 = predict(session_5, image) if model_mode != "Solo 3 clases" else None

    st.subheader("Resultados")
    cols = st.columns(2)

    if probs_3 is not None:
        pred_idx_3 = int(np.argmax(probs_3))
        pred_3 = labels_3[pred_idx_3]
        conf_3 = float(probs_3[pred_idx_3])
        conf_txt_3, conf_class_3 = confidence_badge(conf_3)

        with cols[0]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("Modelo 3 clases")
            st.markdown(f"Predicción: {pred_3}")
            st.markdown(
                f"Confianza: {conf_3:.2f} "
                f"<span class='{conf_class_3}'>{conf_txt_3}</span>",
                unsafe_allow_html=True
            )

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
            st.markdown("Modelo 5 clases")
            st.markdown(f"Estadio predicho: {pred_5}")
            st.markdown(
                f"Confianza: {conf_5:.2f} "
                f"<span class='{conf_class_5}'>{conf_txt_5}</span>",
                unsafe_allow_html=True
            )

            if show_probs:
                st.pyplot(plot_probabilities(probs_5, labels_5, "Probabilidades (5 clases)"))
            st.markdown("</div>", unsafe_allow_html=True)



# MODO 2: Múltiples imágenes (carpeta)
else:
    st.subheader("Cargar carpeta de imágenes")
    st.caption("Selecciona varias imágenes a la vez (equivalente a cargar un folder).")

    uploaded_files = st.file_uploader(
        "Sube múltiples imágenes (.jpg, .png)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Carga varias ecografías para generar un reporte.")
        st.stop()

    rows = []
    with st.spinner("Procesando imágenes..."):
        for uf in uploaded_files:
            img = Image.open(uf).convert("RGB")

            probs_3 = predict(session_3, img) if model_mode != "Solo 5 clases" else None
            probs_5 = predict(session_5, img) if model_mode != "Solo 3 clases" else None

            r = {"name": uf.name, "image": img}

            if probs_3 is not None:
                idx3 = int(np.argmax(probs_3))
                r["pred3"] = labels_3[idx3]
                r["conf3"] = float(probs_3[idx3])
                r["probs3"] = probs_3.tolist()
            else:
                r["pred3"] = None
                r["conf3"] = None

            if probs_5 is not None:
                idx5 = int(np.argmax(probs_5))
                r["pred5"] = labels_5[idx5]
                r["conf5"] = float(probs_5[idx5])
                r["probs5"] = probs_5.tolist()
            else:
                r["pred5"] = None
                r["conf5"] = None

            rows.append(r)

    # Galería
    st.subheader("Galería y predicción")
    n_cols = 4
    grid = st.columns(n_cols)

    for i, r in enumerate(rows):
        with grid[i % n_cols]:
            st.image(r["image"], use_container_width=True)
            if r["pred3"] is not None:
                st.write(f"3 clases: {r['pred3']} ({r['conf3']:.2f})")
            if r["pred5"] is not None:
                st.write(f"5 clases: {r['pred5']} ({r['conf5']:.2f})")

    # Tabla resumen
    st.subheader("Reporte resumen")
    table_data = []
    for r in rows:
        table_data.append({
            "Imagen": r["name"],
            "Predicción 3 clases": r["pred3"],
            "Confianza 3 clases": None if r["conf3"] is None else round(r["conf3"], 3),
            "Predicción 5 clases": r["pred5"],
            "Confianza 5 clases": None if r["conf5"] is None else round(r["conf5"], 3),
        })

    st.dataframe(table_data, use_container_width=True)

    # Exportar PDF
    st.subheader("Exportar reporte")
    pdf_buffer = generate_pdf_report(rows)

    st.download_button(
        label="Exportar como PDF",
        data=pdf_buffer,
        file_name="reporte_fibrosis_hepatica.pdf",
        mime="application/pdf"
    )



st.markdown("---")
st.caption(
    "Esta herramienta es un sistema de apoyo a la decisión clínica. "
    "No reemplaza evaluación médica, historia clínica ni pruebas complementarias."
)
