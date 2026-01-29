# Real-Time Object Detection API and Web App

## Overview
This project validates the integration of a YOLOv8 object detection model into a containerized application with a FastAPI backend and a Streamlit frontend.

## Project Structure
- `api/`: FastAPI application code.
- `ui/`: Streamlit web interface.
- `models/`: Stores the YOLOv8 model (not versioned).
- `scripts/`: Helper scripts (e.g., model downloader).
- `output/`: Directory for processing results.

## Requirements
- Docker & Docker Compose

## Setup & Run
1.  **Environment Setup**:
    Copy `.env.example` to `.env`:
    ```bash
    cp .env.example .env
    ```

2.  **Start Services**:
    Build and start the services using Docker Compose:
    ```bash
    docker-compose up --build
    ```
    This command will:
    - Build the API and UI images.
    - Download the YOLOv8 model (if not present).
    - Start the API service on `http://localhost:8000`.
    - Start the UI service on `http://localhost:8501`.

## Features
- **Object Detection API**: `/detect` endpoint accepts images and returns bounding boxes, labels, and confidence scores. Includes a `/health` check.
- **Web Interface**: Upload images, adjust confidence thresholds, and view annotated results visually.
- **Model Persistence**: Models are downloaded via script and persisted using Docker volumes.

## API Documentation
Once running, visit `http://localhost:8000/docs` for the interactive Swagger UI.
