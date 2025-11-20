import streamlit as st
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor


# ---------------------------------------
# Configuración general de la página
# ---------------------------------------
st.set_page_config(
    page_title="Clasificación de Fibrosis Hepática",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("Clasificador de Fibrosis Hepática")
st.write(
    "Sube una imagen y utiliza las pestañas para visualizar las predicciones "
    "del modelo de tres clases y del modelo de cinco clases."
)


# ---------------------------------------
# Cargar preprocesador
# ---------------------------------------
preprocessor_path = "modelos"  # carpeta donde guardaste el AutoImageProcessor
image_processor = AutoImageProcessor.from_pretrained(preprocessor_path)


# ---------------------------------------
# Cargar modelos ONNX
# ---------------------------------------
session_3 = ort.InferenceSession("modelos/3_clases.onnx", providers=["CPUExecutionProvider"])
session_5 = ort.InferenceSession("modelos/5_clases.onnx", providers=["CPUExecutionProvider"])


# ---------------------------------------
# Funciones auxiliares
# ---------------------------------------
def preprocess(image: Image.Image):
    """Aplica el mismo preprocesamiento que usaste en entrenamiento."""
    encoded = image_processor(image, return_tensors="np")
    return encoded["pixel_values"]  # [1, 3, H, W]


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
    """Crea un barplot sencillo y limpio con Matplotlib."""
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(labels, probabilities)
    ax.set_ylim([0, 1])
    ax.set_ylabel("Probabilidad")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    plt.tight_layout()
    return fig


# ---------------------------------------
# Interfaz principal con un solo uploader
# ---------------------------------------
uploaded_file = st.file_uploader("Subir imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Cargar y mostrar la imagen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Ejecutar ambos modelos una sola vez
    probs_3 = predict(session_3, image)
    probs_5 = predict(session_5, image)

    labels_3 = ["Sano", "Fibrosis", "Cirrosis"]
    labels_5 = ["F0", "F1", "F2", "F3", "F4"]

    pred_3 = labels_3[int(np.argmax(probs_3))]
    pred_5 = labels_5[int(np.argmax(probs_5))]

    # Tabs para elegir qué predicción ver
    tab_3, tab_5 = st.tabs(["Modelo 3 clases", "Modelo 5 clases"])

    with tab_3:
        st.subheader("Predicción del modelo de 3 clases")
        st.write(f"Resultado: {pred_3}")
        fig3 = plot_probabilities(probs_3, labels_3, "Distribución de probabilidades (3 clases)")
        st.pyplot(fig3)

    with tab_5:
        st.subheader("Predicción del modelo de 5 clases")
        st.write(f"Resultado: {pred_5}")
        fig5 = plot_probabilities(probs_5, labels_5, "Distribución de probabilidades (5 clases)")
        st.pyplot(fig5)
