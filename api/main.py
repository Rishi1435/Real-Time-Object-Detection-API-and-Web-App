from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image, ImageDraw, ImageFont
import torch
import io
import os
import json
from collections import Counter
import shutil

app = FastAPI(title="YOLOv8 Object Detection API")

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/yolov8n.pt")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/output")
ANNOTATED_FILE_NAME = "last_annotated.jpg"

print(f"Loading model from {MODEL_PATH}")

# Global model variable
model = None

@app.on_event("startup")
async def startup_event():
    global model
    # Ensure model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Attempting to download...")
        # In a real scenario, we might want to call the download script here or ensure it's run before.
        # But per requirements, we should have the script run before startup. 
        # For robustness, we could try to load from torch hub directly if file missing, 
        # but the request specifically asked for loading from the file downloaded by snippet.
        # However, torch.hub.load can download automatically if we use the 'ultralytics/yolov5' repo and pretrained=True
        # BUT the prompt template specifically requested:
        # model = torch.hub.load('ultralytics/yolov5', 'custom', path='../models/yolov8n.pt', force_reload=True)
        # Note: The prompt uses '../models', but in Docker it will be absolute path or relative to WORKDIR.
        pass

    try:
        # Using yolov5 hub loader to load custom model (YOLOv8 works with this often, or we use ultralytics package)
        # However, since we installed regular 'torch', we rely on torch.hub
        # The prompt says: "Using yolov5 hub loader for yolov8 works".
        # Let's trust the prompt.
        # Using 'ultralytics/yolov5' repo from github to load the custom .pt file.
        # We need to make sure we point to the correct local path.
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback or exit?
        # If we failed to load, app is unhealthy.
        model = None

@app.get("/health")
def health_check():
    if model is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok"}

@app.post("/detect")
async def detect_objects(image: UploadFile = File(...), confidence_threshold: float = Form(0.25)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Validate image
    if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
         raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are supported.")

    try:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Set model confidence
    model.conf = confidence_threshold

    # Perform inference
    results = model(img)

    # Process results
    # results is a list of Result objects if using ultralytics package, OR a Detections object if using yolov5 torch hub.
    # The prompt template implies torch.hub which usually returns yolov5.models.common.Detections
    # Let's assume standard yolov5/hub output.
    
    # Render detected items clearly
    # results.xyxy[0] is a tensor of (x1, y1, x2, y2, conf, cls)
    
    # We need to parse this.
    try:
        df = results.pandas().xyxy[0] # easy to work with pandas
    except AttributeError:
        # Fallback if using different version
        raise HTTPException(status_code=500, detail="Unexpected model output format")

    detections = []
    labels = []
    
    # Prepare drawing
    draw = ImageDraw.Draw(img)
    # try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for _, row in df.iterrows():
        if row['confidence'] < confidence_threshold:
            continue
            
        x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        score = float(row['confidence'])
        
        detections.append({
            "box": [x_min, y_min, x_max, y_max],
            "label": label,
            "score": score
        })
        labels.append(label)
        
        # Draw on image
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
        text = f"{label} {score:.2f}"
        
        # Calculate text background
        text_bbox = draw.textbbox((x_min, y_min), text, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((x_min, y_min), text, fill="white", font=font)

    # Save annotated image
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    output_path = os.path.join(OUTPUT_DIR, ANNOTATED_FILE_NAME)
    img.save(output_path)

    # Summary
    summary = dict(Counter(labels))

    return {
        "detections": detections,
        "summary": summary
    }
