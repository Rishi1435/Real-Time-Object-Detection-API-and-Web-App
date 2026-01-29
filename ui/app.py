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
    value=0.25, 
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
                    # Since API saves to disk, we can't easily retrieve the file unless we serve it back.
                    # The prompt requirement says: "On a successful response, it displays the returned annotated image".
                    # However, the API spec says: "on every successful call... save... to specific output directory".
                    # It DOES NOT say the API returns the image in the response body.
                    # The response body is JSON.
                    # If I need to display the annotated image in Streamlit, I have a few options:
                    # 1. The API should ideally return the image or a URL.
                    # 2. Or Streamlit mounts the SAME output volume and reads it.
                    # The prompt verification for UI says: 
                    # "On a successful response, it displays the returned annotated image and the JSON summary."
                    # PLEASE NOTE: "displays the returned annotated image" implies the API returns it OR we fetch it.
                    # Checking requirement 8: "The API must save... to output/last_annotated.jpg".
                    # Checking requirement 9 (UI): "On a successful response, it displays the returned annotated image".
                    # There is a slight disconnect. If API returns JSON only, how does UI get the image?
                    # OPTION A: UI manually draws boxes on its copy of the image using the JSON data.
                    # OPTION B: UI reads from the shared volume `output/last_annotated.jpg`.
                    # OPTION C: Update API to return image? User said "API Design ... Return JSON response" but specifically "Design your API response to be clear...".
                    # BUT Requirement 5 specifies the response JSON schema. It does NOT include the image data.
                    # So Option A (Draw locally) or Option B (Shared Volume) are viable.
                    # Given they are in containers, sharing a volume `output` is the way to go if we want to show "the api's version".
                    # But if multiple users use it, `last_annotated.jpg` would be overwritten. Race condition.
                    # However, for this assignment, "last_annotated.jpg" seems to be a single global debug file.
                    # The BEST UX is Option A: Use the JSON boxes to draw on the frontend or Option B: Shared Volume.
                    # Let's look at `docker-compose.yml` template in prompt.
                    # `models` and `output` are volumes.
                    # `ui` container does NOT have `output` volume mounted in the prompt's `docker-compose.yml` example!!!
                    # Wait, let's re-read the `docker-compose.yml` template in Step 5 of the prompt.
                    # Service `api` has `volumes: - ./output:/app/output`.
                    # Service `ui` has NO volumes.
                    # So `ui` CANNOT see `last_annotated.jpg`.
                    # Conclusion: The UI should probably DRAW the boxes itself based on JSON response, 
                    # OR the prompt implies the API response might return the image but the JSON schema doesn't show it.
                    # "On a successful response, it displays the returned annotated image" -> slightly ambiguous.
                    # Actually, "The image should visually represent the data returned in the JSON response." - referring to the saved file.
                    # Requirement 9: "On a successful response, it displays the returned annotated image".
                    # This might be loose wording for "displays the image with annotations".
                    # I will implement Option A: Draw boxes on the original image using PIL in Streamlit.
                    # This is robust and stateless. 
                    
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

