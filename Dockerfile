# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV and other common libraries
# NOTE: libgl1-mesa-glx no longer exists in Debian Trixie
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-gl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port 5000 for Flask API
EXPOSE 5000

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:5000/ || exit 1

# Start the app
CMD ["python", "apps/flask_server.py"]
