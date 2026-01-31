import streamlit as st
import requests
from PIL import Image
import os
import io

st.set_page_config(page_title="YOLOv8 Object Detection", page_icon="📷")

st.title("YOLOv8 Object Detection")
st.write("Upload an image to detect objects using YOLOv8.")

# Get API URL from environment variable
API_URL = os.getenv("API_URL", "http://api:8000/detect")

# Sidebar configuration
st.sidebar.header("Configuration")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.40, 
    step=0.05
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    
    st.write("")
    
    if st.button("Detect Objects"):
        with st.spinner("Detecting..."):
            try:
                # Reset file pointer to beginning
                uploaded_file.seek(0)
                files = {"image": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")}
                data = {"confidence_threshold": confidence_threshold}
                
                response = requests.post(API_URL, files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    detections = result.get("detections", [])
                    summary = result.get("summary", {})
                    
                    st.success("Detection complete!")
                    
                    # Display summary
                    st.subheader("Summary")
                    if summary:
                        for label, count in summary.items():
                            st.write(f"- **{label}**: {count}")
                    else:
                        st.write("No objects detected.")
                        
                    # Display detected objects details
                    with st.expander("Detailed Detections"):
                        st.json(detections)
                        
                    # Display annotated image if possible
                    # We draw the boxes locally using the JSON response for a stateless experience. 
                    
                    # Drawing logic
                    if detections:
                        img_draw = Image.open(uploaded_file)
                        from PIL import ImageDraw, ImageFont
                        draw = ImageDraw.Draw(img_draw)
                        try:
                            font = ImageFont.truetype("arial.ttf", 15)
                        except IOError:
                            font = ImageFont.load_default()
                            
                        for det in detections:
                            box = det['box']
                            label = det['label']
                            score = det['score']
                            
                            # Draw box
                            draw.rectangle(box, outline="red", width=3)
                            
                            # Draw label
                            text = f"{label} {score:.2f}"
                            text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
                            draw.rectangle(text_bbox, fill="red")
                            draw.text((box[0], box[1]), text, fill="white", font=font)
                        
                        st.image(img_draw, caption="Annotated Image", use_container_width=True)
                    else:
                        st.image(uploaded_file, caption="No objects detected", use_container_width=True)

                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except requests.exceptions.ConnectionError:
                 st.error("Could not connect to API. Is it running?")
            except Exception as e:
                st.error(f"An error occurred: {e}")

