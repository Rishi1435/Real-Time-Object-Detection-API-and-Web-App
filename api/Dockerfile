FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (e.g., for OpenCV or curl/wget if needed)
# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=3000 --retries 20 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --default-timeout=3000 --retries 20 -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Command to run (will be overridden or augmented by docker-compose)
# We assume the model download script is available via volume or copy
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
