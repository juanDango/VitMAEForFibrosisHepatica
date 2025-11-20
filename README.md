# Predicción de Fibrosis Hepática

Aplicación web desarrollada con Streamlit para predecir el nivel de fibrosis hepática utilizando modelos de deep learning en formato ONNX.

## Descripción

Esta aplicación permite cargar imágenes médicas y predecir el nivel de fibrosis hepática utilizando dos modelos diferentes:

- **Modelo de 3 clases**: Clasifica en Sano, Fibrosis o Cirrosis
- **Modelo de 5 clases**: Clasifica en F0, F1, F2, F3 o F4 (escala METAVIR)

## Instalación

### Requisitos previos
- Python 3.12 o superior
- [uv](https://docs.astral.sh/uv/) - Gestor de paquetes ultrarrápido para Python

### Instalación de uv

Si aún no tienes `uv` instalado, puedes instalarlo de varias formas:

**Opción 1: Con pip (recomendado si ya tienes Python)**
```bash
pip install uv
```

**Opción 2: Instalador standalone**

macOS/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Pasos de instalación del proyecto

1. Clona o descarga este repositorio

2. Navega al directorio del proyecto:
```bash
cd fibrosis
```

3. Instala las dependencias con uv:
```bash
uv sync
```

Esto creará automáticamente un entorno virtual y instalará todas las dependencias especificadas en [`pyproject.toml`](pyproject.toml:1).

### Instalación alternativa (sin uv)

Si prefieres usar pip tradicional:
```bash
pip install -e .
```

## Uso

### Con uv (recomendado)

Para ejecutar la aplicación usando uv:

```bash
uv run streamlit run app.py
```

### Con pip tradicional

Si instalaste con pip:

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## Características

- **Carga de imágenes**: Soporta formatos PNG, JPG y JPEG
- **Visualización**: Muestra la imagen original y las dimensiones
- **Predicción dual**: Ejecuta ambos modelos simultáneamente
- **Selección de modelo**: Permite elegir qué resultados visualizar (3 o 5 clases)
- **Distribución de probabilidades**: Gráficos de barras con Matplotlib
- **Resultados detallados**: Tabla con probabilidades exactas de cada clase
- **Comparación**: Muestra también los resultados del otro modelo

## Modelos

### Modelo de 3 Clases
- **Clase 0**: Sano
- **Clase 1**: Fibrosis
- **Clase 2**: Cirrosis

### Modelo de 5 Clases (Escala METAVIR)
- **F0**: Sin fibrosis
- **F1**: Fibrosis leve (portal)
- **F2**: Fibrosis moderada (periportal)
- **F3**: Fibrosis severa (septos)
- **F4**: Cirrosis

## Estructura del Proyecto

```
fibrosis/
├── app.py                          # Aplicación principal de Streamlit
├── main.py                         # Script principal (legacy)
├── pyproject.toml                  # Configuración y dependencias
├── README.md                       # Este archivo
└── modelos/
    ├── 3_clases.onnx              # Modelo ONNX de 3 clases
    ├── 5_clases.onnx              # Modelo ONNX de 5 clases
    └── preprocessor_config.json    # Configuración del preprocesador
```

## Configuración del Preprocesador

El archivo `preprocessor_config.json` contiene los parámetros de preprocesamiento:
- Redimensionamiento a 224x224 píxeles
- Normalización con media [0.485, 0.456, 0.406]
- Desviación estándar [0.229, 0.224, 0.225]
- Factor de reescalado: 1/255

## Notas Técnicas

- Los modelos están en formato ONNX para inferencia rápida
- Se utiliza `onnxruntime` para la ejecución de los modelos
- Las imágenes se preprocesan según la configuración del ViT Image Processor
- Las probabilidades se calculan aplicando softmax a los logits de salida

## Advertencia

Esta aplicación es solo para fines educativos y de investigación. No debe utilizarse como herramienta de diagnóstico médico sin la supervisión de profesionales de la salud calificados.

## Licencia

Este proyecto es parte de un trabajo académico de la Maestría en Deep Learning.