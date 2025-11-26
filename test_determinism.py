"""Script para probar el determinismo de los modelos ONNX"""
import onnxruntime as ort
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor

# Cargar preprocesador y modelos
preprocessor_path = "modelos"
image_processor = AutoImageProcessor.from_pretrained(preprocessor_path)

session_3 = ort.InferenceSession("modelos/3_clases.onnx", providers=["CPUExecutionProvider"])
session_5 = ort.InferenceSession("modelos/5_clases.onnx", providers=["CPUExecutionProvider"])

def preprocess(image: Image.Image):
    """Aplica el mismo preprocesamiento que usaste en entrenamiento."""
    encoded = image_processor(image, return_tensors="np")
    return encoded["pixel_values"]

def predict(session: ort.InferenceSession, image: Image.Image):
    """Ejecuta inferencia sobre una imagen usando una sesión ONNX."""
    inputs = preprocess(image)
    
    # Imprimir información de debug
    print(f"Input shape: {inputs.shape}")
    print(f"Input dtype: {inputs.dtype}")
    print(f"Input min/max: {inputs.min():.6f} / {inputs.max():.6f}")
    print(f"Input mean/std: {inputs.mean():.6f} / {inputs.std():.6f}")
    
    output_metadata = session.get_outputs()
    output_names = [o.name for o in output_metadata]
    outputs = session.run(output_names, {"pixel_values": inputs})
    
    if "probabilities" in output_names:
        idx = output_names.index("probabilities")
        probs = outputs[idx][0]
    elif "logits" in output_names:
        idx = output_names.index("logits")
        logits = outputs[idx][0]
        print(f"Logits: {logits}")
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
    else:
        raise RuntimeError(f"Salidas disponibles: {output_names}")
    
    return probs

# Cargar una imagen de prueba
print("Carga una imagen de prueba (ej: test.jpg)")
image_path = input("Ruta de la imagen: ")
image = Image.open(image_path).convert("RGB")

print("\n" + "="*60)
print("PRUEBA DE DETERMINISMO - 5 EJECUCIONES")
print("="*60)

# Ejecutar 5 veces el modelo de 3 clases
print("\n--- MODELO 3 CLASES ---")
for i in range(5):
    print(f"\nEjecución {i+1}:")
    probs = predict(session_3, image)
    print(f"Probabilidades: {probs}")
    print(f"Predicción: {np.argmax(probs)} (confianza: {probs[np.argmax(probs)]:.6f})")

# Ejecutar 5 veces el modelo de 5 clases
print("\n--- MODELO 5 CLASES ---")
for i in range(5):
    print(f"\nEjecución {i+1}:")
    probs = predict(session_5, image)
    print(f"Probabilidades: {probs}")
    print(f"Predicción: {np.argmax(probs)} (confianza: {probs[np.argmax(probs)]:.6f})")