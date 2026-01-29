import requests
import time
import os
import json

API_URL = "http://localhost:8000"
DETECT_URL = f"{API_URL}/detect"
HEALTH_URL = f"{API_URL}/health"
IMAGE_PATH = "test_image.jpg"
OUTPUT_FILE = "output/last_annotated.jpg"

def wait_for_health():
    print("Waiting for API health...")
    for _ in range(30):
        try:
            r = requests.get(HEALTH_URL)
            if r.status_code == 200:
                print("API is healthy!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    print("API failed to become healthy.")
    return False

def test_detect():
    print(f"Testing detection with {IMAGE_PATH}...")
    if not os.path.exists(IMAGE_PATH):
        print("Test image not found.")
        return False
    
    with open(IMAGE_PATH, "rb") as f:
        files = {"image": ("test_image.jpg", f, "image/jpeg")}
        data = {"confidence_threshold": 0.25}
        response = requests.post(DETECT_URL, files=files, data=data)
    
    if response.status_code == 200:
        print("Detection success!")
        result = response.json()
        print(json.dumps(result, indent=2))
        
        # Verify schema
        if "detections" in result and "summary" in result:
            print("Response schema valid.")
        else:
            print("Response schema invalid.")
            return False
            
        # Verify content
        if len(result["detections"]) > 0:
            print(f"Detected objects: {result['summary']}")
        else:
            print("No objects detected (unexpected for this image).")
            
        return True
    else:
        print(f"Detection failed: {response.status_code} {response.text}")
        return False

def check_output_file():
    print(f"Checking for output file {OUTPUT_FILE}...")
    # Note: This checks the host volume mount.
    if os.path.exists(OUTPUT_FILE):
        print("Output file exists.")
        return True
    else:
        print("Output file missing.")
        return False

if __name__ == "__main__":
    if wait_for_health():
        if test_detect():
            if check_output_file():
                print("ALL TESTS PASSED")
                exit(0)
    print("TESTS FAILED")
    exit(1)
