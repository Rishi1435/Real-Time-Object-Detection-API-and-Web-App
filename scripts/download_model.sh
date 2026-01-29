#!/bin/bash
# Create the models directory if it doesn't exist
mkdir -p models

# URL of the YOLOv8n model
MODEL_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"

# Download the model using wget or curl
if command -v wget &> /dev/null; then
    wget -O models/yolov8n.pt $MODEL_URL
elif command -v curl &> /dev/null; then
    curl -L -o models/yolov8n.pt $MODEL_URL
else
    echo "Error: Neither wget nor curl is installed. Cannot download model."
    exit 1
fi

echo "Model downloaded successfully to models/yolov8n.pt"
