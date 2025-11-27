import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from transformers import ViTMAEModel
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO
from datetime import datetime
import random
import os

# Librerías de PDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ==============================================================================
# 1. LÓGICA DE DETERMINISMO (El "Candado")
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
# 2. CONFIGURACIÓN Y RUTAS
# ==============================================================================
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "modelos"

# Rutas a tus archivos .pth (Ajusta los nombres si es necesario)
MODEL_3_PATH = MODELS_DIR / "vit_classifier_3_clases.pth"
MODEL_5_PATH = MODELS_DIR / "vit_classifier_5_clases.pth"

st.set_page_config(
    page_title="Clasificador de Fibrosis Hepática por Ecografía",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Estilizado
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.hero {
    background: linear-gradient(90deg, #0ea5e9 0%, #22c55e 100%);
    padding: 1.5rem 2rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.hero h1 {font-size: 2.2rem; margin: 0; font-weight: 700;}
.hero p {font-size: 1.1rem; opacity: 0.95; margin-top: 0.5rem;}
.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    height: 100%;
}
.badge {
    display:inline-block; padding:0.25rem 0.75rem; border-radius:9999px;
    font-size:0.875rem; font-weight:600;
}
.badge-high {background:#dcfce7; color:#166534;}
.badge-med {background:#fef9c3; color:#854d0e;}
.badge-low {background:#fee2e2; color:#991b1b;}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. DEFINICIÓN DEL MODELO PYTORCH (Backend Robusto)
# ==============================================================================
class ViTMAEForFibrosisClassification(nn.Module):
    def __init__(self, num_labels, encoder):
        super().__init__()
        self.encoder = encoder
        # Arquitectura exacta de tu entrenamiento
        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.4), # Se desactiva con model.eval()
            nn.Linear(128, num_labels)
        )

    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values)
        sequence_output = outputs.last_hidden_state
        cls_token = sequence_output[:, 0, :]
        logits = self.classifier(cls_token)
        return logits

# ==============================================================================
# 4. CARGA DE MODELOS Y PREPROCESAMIENTO (Cacheado)
# ==============================================================================

@st.cache_resource(show_spinner=False)
def load_pytorch_model(weights_path, num_labels):
    """Carga robusta en CPU con PyTorch nativo."""
    device = torch.device("cpu")
    
    # 1. Base Encoder
    try:
        base_encoder = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
    except Exception as e:
        st.error(f"Error conectando con HuggingFace: {e}")
        return None
        
    # 2. Estructura
    model = ViTMAEForFibrosisClassification(num_labels, encoder=base_encoder)
    
    # 3. Pesos
    path_str = str(weights_path)
    if not os.path.exists(path_str):
        return None # Se maneja en la UI
        
    try:
        state_dict = torch.load(path_str, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        st.error(f"Error cargando pesos {path_str}: {e}")
        return None

    # 4. Configuración Eval
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    return model

# Preprocesamiento Manual (Tu JSON traducido a Torchvision)
def get_deterministic_transform():
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# Cargar modelos al inicio
with st.spinner("Inicializando motor de IA..."):
    model_3 = load_pytorch_model(MODEL_3_PATH, 3)
    model_5 = load_pytorch_model(MODEL_5_PATH, 5)

# ==============================================================================
# 5. LÓGICA DE INFERENCIA
# ==============================================================================
def predict_torch(model, image: Image.Image):
    """Inferencia determinista en PyTorch."""
    if model is None:
        return None
        
    # Preprocesar
    pipeline = get_deterministic_transform()
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    input_tensor = pipeline(image).unsqueeze(0) # [1, 3, 224, 224]
    
    # Inferencia
    seed_everything(42) # Candado extra
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
        
    return probs

def plot_probabilities(probabilities, labels, title):
    fig, ax = plt.subplots(figsize=(6, 3))
    # Colores suaves
    ax.barh(labels, probabilities, color='#0ea5e9', alpha=0.8)
    ax.set_xlim([0, 1])
    ax.set_xlabel("Probabilidad")
    ax.set_title(title, fontsize=10, pad=10)
    
    # Quitar bordes feos
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for i, v in enumerate(probabilities):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=9, fontweight='bold')
    plt.tight_layout()
    return fig

def confidence_badge(p):
    if p >= 0.75:
        return "Alta", "badge badge-high"
    if p >= 0.50:
        return "Media", "badge badge-med"
    return "Baja", "badge badge-low"

# ==============================================================================
# 6. GENERADOR DE PDF (ReportLab)
# ==============================================================================
def generate_pdf_report(rows, title="Reporte Fibrosis Hepática"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 40
    y = height - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, title)
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 40

    for i, r in enumerate(rows, start=1):
        if y < 250:
            c.showPage()
            y = height - margin

        # Imagen
        img_pil = r["image"].copy()
        img_pil.thumbnail((200, 200))
        img_io = BytesIO()
        img_pil.save(img_io, format="PNG")
        img_io.seek(0)
        
        c.drawImage(ImageReader(img_io), margin, y - 120, width=120, height=120, preserveAspectRatio=True, mask='auto')
        
        # Textos
        text_x = margin + 140
        c.setFont("Helvetica-Bold", 12)
        c.drawString(text_x, y - 20, f"Imagen {i}: {r['name']}")
        
        c.setFont("Helvetica", 10)
        curr_text_y = y - 45
        
        if r.get("pred3"):
            c.drawString(text_x, curr_text_y, f"Modelo 3 clases: {r['pred3']} (Conf: {r['conf3']:.2f})")
            curr_text_y -= 15
        
        if r.get("pred5"):
            c.drawString(text_x, curr_text_y, f"Modelo 5 clases: {r['pred5']} (Conf: {r['conf5']:.2f})")
        
        y -= 160

    c.save()
    buffer.seek(0)
    return buffer

# ==============================================================================
# 7. INTERFAZ DE USUARIO (Frontend)
# ==============================================================================

# Header Hero
st.markdown("""
<div class="hero">
  <h1>Clasificador de Fibrosis Hepática por Ecografía</h1>
  <p>
    Sistema de apoyo clínico basado en Vision Transformers (ViT-MAE) con ejecución determinística.
  </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Configuración")
input_mode = st.sidebar.radio("Modo de entrada", ["Imagen única", "Múltiples imágenes (Lote)"])
show_probs = st.sidebar.checkbox("Mostrar gráficas", value=True)

st.sidebar.markdown("---")
st.sidebar.info("Ambos modelos ejecutándose en CPU con PyTorch Native.")

labels_3 = ["Sano", "Fibrosis", "Cirrosis"]
labels_5 = ["F0", "F1", "F2", "F3", "F4"]

# --- MODO 1: IMAGEN ÚNICA ---
if input_mode == "Imagen única":
    st.subheader("Análisis Individual")
    uploaded_file = st.file_uploader("Sube una ecografía", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        
        col_viz, col_res = st.columns([1, 1.5])
        
        with col_viz:
            st.image(image, caption="Imagen cargada", use_column_width=True)
            # Checksum discreto para verificar determinismo si lo necesitas
            # st.caption(f"ID Imagen: {np.array(image).sum()}") 

        with col_res:
            # Ejecución
            probs_3 = predict_torch(model_3, image)
            probs_5 = predict_torch(model_5, image)
            
            # Mostrar resultados
            tab3, tab5 = st.tabs(["Diagnóstico (3 Clases)", "Estadio (5 Clases)"])
            
            with tab3:
                idx = np.argmax(probs_3)
                lbl = labels_3[idx]
                conf = probs_3[idx]
                txt, badge = confidence_badge(conf)
                
                st.markdown(f"### Predicción: **{lbl}**")
                st.markdown(f"Confianza: <span class='{badge}'>{txt} ({conf:.1%})</span>", unsafe_allow_html=True)
                if show_probs:
                    st.pyplot(plot_probabilities(probs_3, labels_3, ""))

            with tab5:
                idx = np.argmax(probs_5)
                lbl = labels_5[idx]
                conf = probs_5[idx]
                txt, badge = confidence_badge(conf)
                
                st.markdown(f"### Estadio: **{lbl}**")
                st.markdown(f"Confianza: <span class='{badge}'>{txt} ({conf:.1%})</span>", unsafe_allow_html=True)
                if show_probs:
                    st.pyplot(plot_probabilities(probs_5, labels_5, ""))

# --- MODO 2: MÚLTIPLES IMÁGENES ---
else:
    st.subheader("Análisis por Lotes")
    uploaded_files = st.file_uploader("Sube múltiples ecografías", type=["jpg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        rows = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            img = Image.open(file).convert("RGB")
            
            p3 = predict_torch(model_3, img)
            p5 = predict_torch(model_5, img)
            
            idx_3 = np.argmax(p3)
            idx_5 = np.argmax(p5)
            
            row = {
                "name": file.name,
                "image": img,
                "pred3": labels_3[idx_3],
                "conf3": float(p3[idx_3]),
                "pred5": labels_5[idx_5],
                "conf5": float(p5[idx_5])
            }
                
            rows.append(row)
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        # Tabla resumen
        st.write("### Resumen de Resultados")
        st.dataframe([
            {
                "Archivo": r["name"],
                "3 Clases": r["pred3"],
                "Conf. 3": f"{r['conf3']:.2%}",
                "5 Clases": r["pred5"],
                "Conf. 5": f"{r['conf5']:.2%}"
            }
            for r in rows
        ], use_container_width=True)
        
        # PDF Download
        pdf_data = generate_pdf_report(rows)
        st.download_button(
            "📄 Descargar Reporte PDF", 
            data=pdf_data, 
            file_name="reporte_fibrosis.pdf", 
            mime="application/pdf"
        )