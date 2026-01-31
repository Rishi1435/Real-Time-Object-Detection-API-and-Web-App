from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image, ImageDraw, ImageFont
import io
import os
import json
from collections import Counter
from ultralytics import YOLO

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
    try:
        if not os.path.exists(MODEL_PATH):
             print(f"Model file not found at {MODEL_PATH} during startup (will attempt load anyway if lib supports auto-download or fail).")
        
        # Load model using ultralytics
        model = YOLO(MODEL_PATH) 
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}

@app.post("/detect")
async def detect_objects(image: UploadFile = File(...), confidence_threshold: float = Form(0.40)):
    if model is None:
         raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate image
    if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
         raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are supported.")

    try:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Perform inference
    # YOLOv8 call
    results = model(img, conf=confidence_threshold)

    detections = []
    labels = []
    
    # Prepare drawing
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # Process results - ultralytics returns a list of Results objects
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # box.xyxy is a tensor of shape (1, 4)
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            
            # box.conf is a tensor of shape (1,)
            score = float(box.conf[0])
            
            # box.cls is a tensor of shape (1,)
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            
            detections.append({
                "box": [x_min, y_min, x_max, y_max],
                "label": label,
                "score": score
            })
            labels.append(label)
            
            # Draw on image
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
            text = f"{label} {score:.2f}"
            
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
