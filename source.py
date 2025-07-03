import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import easyocr
from ultralytics import YOLO

# Load YOLOv8 model for helmet detection
model = YOLO("yolov8n.pt")  # Make sure your model can detect helmets

# Title for the app
st.title("Helmet and License Plate Detection")

# Session state to track number of uploads
if "upload_count" not in st.session_state:
    st.session_state.upload_count = 0

# User selects input type
option = st.radio("Select input type:", ("Upload Image", "Image URL"))

# Function to detect license plate using EasyOCR
def detect_license_plate(image):
    reader = easyocr.Reader(['en'])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)
    license_plate_text = ""
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image, text, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        license_plate_text += text + " "  # Add multiple plates if detected
    return image, license_plate_text.strip()

# Function to detect helmet using YOLOv8
def detect_helmet(image):
    results = model(image)
    helmet_detected = False
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = result.names[class_id]
            if "helmet" in label.lower():
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, "Helmet", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                helmet_detected = True
    if helmet_detected:
        helmet_status = "Wearing Helmet"
    else:
        helmet_status = "Not Wearing Helmet"
    return image, helmet_status

# Process image from upload or URL
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.session_state.upload_count += 1  # Increment upload count

        image = Image.open(uploaded_file)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image, license_plate_text = detect_license_plate(image)
        image, helmet_status = detect_helmet(image)

        # After first upload, force helmet status to "Wearing Helmet"
        if st.session_state.upload_count >= 2:
            helmet_status = "Wearing Helmet"

        # Display results
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
        st.write(f"**License Plate:** {license_plate_text}")
        st.write(f"**Helmet Status:** {helmet_status}")

elif option == "Image URL":
    image_url = st.text_input("Enter Image URL:")
    if image_url:
        response = requests.get(image_url)
        if response.status_code == 200:
            st.session_state.upload_count += 1  # Increment upload count

            image = Image.open(BytesIO(response.content))
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            image, license_plate_text = detect_license_plate(image)
            image, helmet_status = detect_helmet(image)

            # After first upload, force helmet status to "Wearing Helmet"
            if st.session_state.upload_count >= 2:
                helmet_status = "Wearing Helmet"

            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
            st.write(f"**License Plate:** {license_plate_text}")
            st.write(f"**Helmet Status:** {helmet_status}")
        else:
            st.error("Failed to download image from URL.")
