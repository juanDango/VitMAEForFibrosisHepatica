import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from transformers import ViTMAEModel
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# ==============================================================================
# 1. CANDADO DE DETERMINISMO (Para que nada sea al azar)
# ==============================================================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# ==============================================================================
# 2. DEFINICIÓN DE LA CLASE DEL MODELO
# ==============================================================================
class ViTMAEForFibrosisClassification(nn.Module):
    def __init__(self, num_labels, encoder):
        super().__init__()
        self.encoder = encoder
        # Tu arquitectura exacta basada en tus logs
        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.4), # Se apaga automáticamente con model.eval()
            nn.Linear(128, num_labels)
        )

    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values)
        sequence_output = outputs.last_hidden_state
        # Usamos el token [CLS] (índice 0)
        cls_token = sequence_output[:, 0, :]
        logits = self.classifier(cls_token)
        return logits

# ==============================================================================
# 3. CONFIGURACIÓN DE PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Clasificación de Fibrosis Hepática",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("Clasificador de Fibrosis Hepática")
st.write("Versión Determinística: PyTorch Nativo + Torchvision Transforms")

# ==============================================================================
# 4. CARGA DE MODELOS
# ==============================================================================
@st.cache_resource
def load_model_robust(weights_path, num_labels):
    device = torch.device("cpu")
    
    # 1. Crear el esqueleto del modelo
    try:
        base_encoder = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
    except Exception as e:
        st.error(f"Error descargando vit-mae-base. Verifica tu internet: {e}")
        return None
        
    model = ViTMAEForFibrosisClassification(num_labels=num_labels, encoder=base_encoder)
    
    # 2. Cargar los pesos entrenados
    if not os.path.exists(weights_path):
        st.error(f"❌ No se encontró el archivo: {weights_path}")
        return None
        
    try:
        # map_location='cpu' evita errores si entrenaste en GPU
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        st.error(f"❌ Error de arquitectura en {weights_path}. Las capas no coinciden: {e}")
        return None

    # 3. MODO EVALUACIÓN (CRÍTICO)
    model.to(device)
    model.eval() 
    
    # Congelar gradientes para ahorrar memoria y evitar cambios
    for param in model.parameters():
        param.requires_grad = False
        
    return model

# --- RUTAS DE TUS ARCHIVOS .PTH (MODIFICA ESTO SI ES NECESARIO) ---
# Asumo que están en la carpeta 'modelos' dentro de tu proyecto
PATH_3_CLASES = "modelos/vit_classifier_3_clases.pth" 
PATH_5_CLASES = "modelos/vit_classifier_5_clases.pth"

with st.spinner("Cargando modelos en memoria..."):
    model_3 = load_model_robust(PATH_3_CLASES, 3)
    model_5 = load_model_robust(PATH_5_CLASES, 5)

if model_3 and model_5:
    st.success("✅ Modelos cargados correctamente en modo EVAL.")

# ==============================================================================
# 5. PREPROCESAMIENTO MANUAL (TRADUCCIÓN DE TU JSON)
# ==============================================================================
def get_transforms_from_json():
    """
    Replica EXACTAMENTE tu preprocessor.json usando Torchvision.
    - Resize a (224, 224) forzado (sin crop).
    - Interpolación Bilinear (resample=2).
    - Normalización ImageNet.
    """
    return transforms.Compose([
        # do_resize: True, size: {h:224, w:224}, resample: 2 (BILINEAR)
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        
        # do_rescale: True (ToTensor divide por 255 automáticamente)
        transforms.ToTensor(),
        
        # do_normalize: True (Tus valores del json)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# ==============================================================================
# 6. FUNCIÓN DE PREDICCIÓN Y GRÁFICOS
# ==============================================================================
def predict_fn(model, image_tensor):
    # Aseguramos determinismo en tiempo de inferencia
    seed_everything(42)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
    return probs.squeeze().numpy()

def plot_probabilities(probabilities, labels, title):
    fig, ax = plt.subplots(figsize=(5, 3))
    colors = ['#4CAF50' if x == max(probabilities) else '#B0BEC5' for x in probabilities]
    ax.bar(labels, probabilities, color=colors)
    ax.set_ylim([0, 1])
    ax.set_ylabel("Probabilidad")
    ax.set_title(title)
    
    for i, v in enumerate(probabilities):
        ax.text(i, v + 0.02, f"{v:.1%}", ha='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    return fig

# ==============================================================================
# 7. INTERFAZ DE USUARIO
# ==============================================================================
uploaded_file = st.file_uploader("Subir imagen de biopsia", type=["jpg", "jpeg", "png"])

if uploaded_file and model_3 and model_5:
    # A. Cargar Imagen
    image = Image.open(uploaded_file).convert("RGB")
    
    col_img, col_data = st.columns([1, 2])
    with col_img:
        st.image(image, caption="Imagen Original", use_column_width=True)

    # B. Preprocesar
    preprocess_pipeline = get_transforms_from_json()
    input_tensor = preprocess_pipeline(image).unsqueeze(0) # Batch size 1
    
    # C. Verificar Estabilidad (Checksum)
    # Si este número cambia al recargar, el problema es la librería de imágenes, no el modelo.
    checksum = input_tensor.sum().item()
    with col_data:
        st.info("Datos de Control (Deben ser fijos al recargar)")
        st.code(f"Tensor Checksum: {checksum:.6f}", language="text")

    # D. Inferencia
    probs_3 = predict_fn(model_3, input_tensor)
    probs_5 = predict_fn(model_5, input_tensor)

    # E. Mostrar Resultados
    labels_3 = ["Sano", "Fibrosis", "Cirrosis"]
    labels_5 = ["F0", "F1", "F2", "F3", "F4"]

    pred_3_idx = np.argmax(probs_3)
    pred_5_idx = np.argmax(probs_5)

    tab_3, tab_5 = st.tabs(["🩺 Modelo 3 Clases", "🔬 Modelo 5 Clases"])

    with tab_3:
        st.subheader(f"Diagnóstico: {labels_3[pred_3_idx]}")
        st.pyplot(plot_probabilities(probs_3, labels_3, "Distribución de Confianza"))

    with tab_5:
        st.subheader(f"Estadio METAVIR: {labels_5[pred_5_idx]}")
        st.pyplot(plot_probabilities(probs_5, labels_5, "Distribución de Confianza"))