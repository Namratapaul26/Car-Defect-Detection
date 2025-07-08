import streamlit as st
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import os
import cv2
import random
from ultralytics import YOLO
import time

# Set page config
st.set_page_config(page_title="Car Defect Detection", layout="wide")
st.title("Car Defect Detection System")

# Initialize session state for webcam control
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False
if 'latest_frame' not in st.session_state:
    st.session_state.latest_frame = None
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = []

# Load a pre-trained YOLOv8 model
# This model is trained on the COCO dataset and can detect 80 common objects.
# The model will be downloaded automatically on the first run.
try:
    # Try to load custom trained model first, fallback to pre-trained
    custom_model_path = 'best.pt'
    if os.path.exists(custom_model_path):
        model = YOLO(custom_model_path)
        st.sidebar.success("✅ Using custom trained model")
    else:
        model = YOLO('yolov8n.pt')
        st.sidebar.warning("⚠️ Using pre-trained model (not custom)")
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    st.stop()


# --- COCO Dataset Sample Logic (Existing Code) ---
# This part of the code remains to show samples from your original dataset.
@st.cache_data
def load_coco_annotations():
    try:
        dataDir = 'C:/Users/STUDIO PC/OneDrive/Desktop/CARDEFECTPROJECT/archive/val'
        dataType = 'COCO_val_annos'
        mul_dataType = 'COCO_mul_val_annos'
        annFile = os.path.join(dataDir, f'{dataType}.json')
        mul_annFile = os.path.join(dataDir, f'{mul_dataType}.json')
        img_dir = "C:/Users/STUDIO PC/OneDrive/Desktop/CARDEFECTPROJECT/archive/img"

        coco = COCO(annFile)
        mul_coco = COCO(mul_annFile)
        return coco, mul_coco, img_dir
    except Exception as e:
        st.error(f"Error loading COCO annotations: {str(e)}")
        st.stop()

coco, mul_coco, img_dir = load_coco_annotations()

cats = coco.loadCats(coco.getCatIds())
mul_cats = mul_coco.loadCats(mul_coco.getCatIds())
damage_categories = {cat['id']: cat['name'] for cat in cats}
part_categories = {cat['id']: cat['name'] for cat in mul_cats}

def get_random_sample():
    """Get a random sample from the COCO dataset"""
    catIds = coco.getCatIds(catNms=['damage'])
    imgIds = coco.getImgIds(catIds=catIds)
    if not imgIds:
        return None, None, None, None
    random_img_id = random.choice(imgIds)
    imgId = coco.getImgIds(imgIds=[random_img_id])
    img_info = coco.loadImgs(imgId)[0]
    img_path = os.path.join(img_dir, img_info['file_name'])
    
    annIds = coco.getAnnIds(imgIds=imgId, iscrowd=None)
    damage_anns = coco.loadAnns(annIds)
    mul_annIds = mul_coco.getAnnIds(imgIds=imgId, iscrowd=None)
    part_anns = mul_coco.loadAnns(mul_annIds)
    
    return img_info, damage_anns, part_anns, img_path

def draw_coco_annotations(img, damage_annotations, part_annotations):
    """Draw COCO annotations on the image"""
    img_copy = img.copy()
    # Draw damage annotations in red
    for ann in damage_annotations:
        bbox = ann['bbox']
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img_copy, "Damage", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    # Draw part annotations in green
    for ann in part_annotations:
        bbox = ann['bbox']
        x, y, w, h = [int(v) for v in bbox]
        category_name = part_categories.get(ann['category_id'], 'Part')
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_copy, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img_copy
# --- End of COCO Sample Logic ---


# --- New YOLOv8 Inference Logic ---
def run_inference(image):
    """Runs YOLOv8 model on the image and returns the annotated image."""
    results = model(image)  # Run inference
    result = results[0]
    
    # Use the plot() method from ultralytics to get an annotated image array
    annotated_img_bgr = result.plot()
    
    # Convert from BGR to RGB for Streamlit display
    annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
    
    # Get the names of detected objects
    detected_objects = [result.names[int(cls)] for cls in result.boxes.cls]
    
    return annotated_img_rgb, detected_objects
# --- End of YOLOv8 Logic ---

# --- Real-time Webcam Functions ---
def process_webcam_frame(frame):
    """Process a single webcam frame"""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run inference
    annotated_frame, detected_objects = run_inference(frame_rgb)
    
    return annotated_frame, detected_objects

def capture_and_process_frame():
    """Capture a single frame and process it"""
    if not st.session_state.webcam_active:
        return None, []
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam")
        st.session_state.webcam_active = False
        return None, []
    
    try:
        ret, frame = cap.read()
        if ret:
            annotated_frame, detected_objects = process_webcam_frame(frame)
            return annotated_frame, detected_objects
        else:
            st.error("Failed to read from webcam")
            st.session_state.webcam_active = False
            return None, []
    finally:
        cap.release()

def start_webcam_detection():
    """Start real-time webcam detection"""
    st.session_state.webcam_active = True

def stop_webcam_detection():
    """Stop real-time webcam detection"""
    st.session_state.webcam_active = False
    st.session_state.latest_frame = None
    st.session_state.detection_results = []

# --- Main Streamlit Interface ---
st.sidebar.title("Controls")
app_mode = st.sidebar.selectbox("Choose the App Mode",
                                ["Real-time Webcam Detection", "Upload Image Detection", "Show Sample from Dataset"])

if app_mode == "Real-time Webcam Detection":
    st.header("🎥 Real-time Car Defect Detection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Instructions:**")
        st.write("1. Click 'Start Webcam Detection' below")
        st.write("2. Allow camera access when prompted")
        st.write("3. Point camera at a car")
        st.write("4. View real-time detection results")
        st.write("5. Click 'Stop Webcam Detection' to stop")
    
    with col2:
        st.write("**Detection Capabilities:**")
        if os.path.exists(custom_model_path):
            st.write("✅ Custom trained model")
            st.write("✅ Car damage detection")
            st.write("✅ Car parts identification")
            st.write("✅ Real-time processing")
        else:
            st.write("⚠️ Pre-trained model only")
            st.write("⚠️ General object detection")
            st.write("⚠️ Not specialized for car defects")
    
    # Webcam control buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎥 Start Webcam Detection", type="primary", disabled=st.session_state.webcam_active):
            start_webcam_detection()
    
    with col2:
        if st.button("⏹️ Stop Webcam Detection", type="secondary", disabled=not st.session_state.webcam_active):
            stop_webcam_detection()
    
    # Display webcam feed and results
    if st.session_state.webcam_active:
        st.write("**Real-time Car Defect Detection Active**")
        st.write("Point your camera at a car to detect defects and parts.")
        
        # Capture and process button
        if st.button("📸 Capture & Detect", type="primary"):
            with st.spinner("Processing frame..."):
                annotated_frame, detected_objects = capture_and_process_frame()
                if annotated_frame is not None:
                    st.session_state.latest_frame = annotated_frame
                    st.session_state.detection_results = detected_objects
        
        # Display the latest frame if available
        if st.session_state.latest_frame is not None:
            st.image(st.session_state.latest_frame, channels="RGB", use_column_width=True)
            
            # Display detected objects
            if st.session_state.detection_results:
                unique_objects = list(set(st.session_state.detection_results))
                st.success(f"**Detected:** {', '.join(unique_objects)}")
            else:
                st.info("No objects detected")
        else:
            st.info("Click 'Capture & Detect' to start detection")
    
    elif st.session_state.latest_frame is not None:
        # Show the last captured frame when stopped
        st.image(st.session_state.latest_frame, caption="Last captured frame", use_column_width=True)
        if st.session_state.detection_results:
            unique_objects = list(set(st.session_state.detection_results))
            st.success(f"**Last Detection:** {', '.join(unique_objects)}")

elif app_mode == "Upload Image Detection":
    st.header("📁 Upload Image Detection")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Objects"):
            with st.spinner("Running model... Please wait."):
                annotated_img, detected_objects = run_inference(image)
                st.image(annotated_img, caption="Model Detections", use_column_width=True)
                if detected_objects:
                    st.success(f"**Detected Objects:** {', '.join(set(detected_objects))}")
                else:
                    st.warning("No objects detected in the image.")

elif app_mode == "Show Sample from Dataset":
    st.header("📊 Sample from COCO-style Car Defect Dataset")
    if st.button("Load New Sample"):
        img_info, damage_anns, part_anns, img_path = get_random_sample()
        if img_info:
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            st.image(img_rgb, caption=f"Original Sample: {img_info['file_name']}", use_column_width=True)
            
            annotated_img = draw_coco_annotations(img_rgb, damage_anns, part_anns)
            st.image(annotated_img, caption="Annotations from Dataset", use_column_width=True)
        else:
            st.error("Could not load a sample from the dataset.")

# Add information about the model
with st.expander("About this Application"):
    st.write("""
    This application provides three modes for car defect detection:

    **1. Real-time Webcam Detection:**
    - Uses your camera for live detection
    - Processes video frames in real-time
    - Shows detection results instantly
    - Perfect for inspecting cars in real-world scenarios

    **2. Upload Image Detection:**
    - Upload and analyze individual images
    - Good for batch processing or detailed analysis
    - Works with any image format

    **3. Show Sample from Dataset:**
    - Displays pre-annotated images from your training dataset
    - Useful for comparing model predictions with ground truth
    - Shows how the dataset was originally labeled
    """)

with st.expander("Model Information"):
    if os.path.exists(custom_model_path):
        st.success("**Custom Trained Model Active**")
        st.write("This model was trained specifically on your car defect dataset.")
        st.write("It can detect:")
        st.write("- Car damages (scratches, dents, etc.)")
        st.write("- Car parts (headlamp, bumper, hood, door, etc.)")
    else:
        st.warning("**Pre-trained Model Active**")
        st.write("This is a general-purpose model trained on the COCO dataset.")
        st.write("It can detect general objects but may not be optimal for car defects.")
        st.write("Consider training a custom model for better results.")

with st.expander("Next Steps: Training Your Own Model"):
    st.info("""
    To detect specific car defects (like 'scratch', 'dent') on uploaded images, you need to train your own model.
    Since your data is in COCO format, you can easily train a YOLOv8 model on it.
    
    High-level steps:
    1. Organize your dataset into the format expected by YOLO.
    2. Write a YAML configuration file describing your dataset paths and classes.
    3. Use the `ultralytics` library to train the model on your data.
    4. Once trained, replace `'yolov8n.pt'` in this app with the path to your custom model file (e.g., `'path/to/your/best.pt'`).
    """) 