# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV and other common libraries
# libgl1-mesa-glx and libglib2.0-0 are often needed for cv2
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code
# specific to this project structure where apps/streamlit_dashboard.py depends on core/
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Healthcheck to ensure the app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the application
# We run from the root directory so Python path finds 'core'
CMD ["streamlit", "run", "apps/streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
