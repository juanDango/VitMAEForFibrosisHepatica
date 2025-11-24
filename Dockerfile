# Usamos python slim
FROM python:3.9-slim

# 1. Copiamos el binario de uv desde su imagen oficial (mucho más rápido que instalarlo)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Configuración de entorno para Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Le decimos a uv que compile el bytecode para un arranque más rápido
ENV UV_COMPILE_BYTECODE=1 

WORKDIR /app

# 2. Copiamos primero SOLO los archivos de dependencias
# Esto permite que Docker use la caché si no has cambiado dependencias,
# aunque cambies tu código app.py
COPY pyproject.toml uv.lock ./

# 3. Instalamos las dependencias
# --frozen: Asegura que se usen las versiones exactas del lockfile
# --no-dev: No instalamos dependencias de desarrollo
# --system: Instalamos en el python del sistema (evitamos crear un venv dentro del container)
RUN uv sync --frozen --no-dev

# 4. IMPORTANTE: Agregamos el venv al PATH
# uv sync crea un entorno virtual en .venv por defecto.
ENV PATH="/app/.venv/bin:$PATH"

# 5. Copiamos el resto del código
COPY . .

EXPOSE 8501

# Chequeo de salud
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Comando de ejecución
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]