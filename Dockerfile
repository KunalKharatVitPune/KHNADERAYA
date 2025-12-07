# Use official Python runtime
FROM python:3.10-slim

# Make output unbuffered (helpful for logs)
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install system deps (minimal) needed by opencv & typical libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /home/appuser/app

# Copy & install Python dependencies first (cache layer)
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure app files owned by non-root user, then switch
RUN chown -R appuser:appuser /home/appuser/app
USER appuser

# Metadata
EXPOSE 5000

# Healthcheck: App Platform sets $PORT at runtime; default to 5000 locally
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl --fail http://localhost:${PORT:-5000}/health || exit 1

# Start with gunicorn (use PORT env var provided by App Platform)
# Use shell form so $PORT is expanded at runtime
CMD ["sh", "-c", "exec gunicorn -w 4 -b 0.0.0.0:${PORT:-5000} apps.flask_server:app"]
