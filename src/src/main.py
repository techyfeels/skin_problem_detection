import numpy as np
import streamlit as st
import supervision as sv
from ultralytics import YOLO
from PIL import Image 
from io import BytesIO
from pathlib import Path
from collections import Counter
from llm_utils import get_skin_advice

# Get project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Cache model loading - models are loaded only once per type
@st.cache_resource
def load_model(model_name: str):
    """Load YOLO model with caching for performance"""
    model_path = MODELS_DIR / f"best_{model_name}.pt"
    return YOLO(str(model_path))

# Cache annotators - reuse same annotator objects
@st.cache_resource
def get_annotators():
    """Get cached annotator objects"""
    return sv.BoxAnnotator(), sv.LabelAnnotator()

def detector_pipeline_pillow(image_bytes, model, conf_threshold=0.25):
    """Optimized detection pipeline"""
    # Load and convert image
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_np_rgb = np.array(pil_image)
    
    # Run inference with confidence threshold
    results = model(image_np_rgb, conf=conf_threshold, verbose=False)[0] 
    detections = sv.Detections.from_ultralytics(results).with_nms()
    
    # Map class IDs to class names from the model
    class_names_dict = model.names
    detected_class_names = [class_names_dict[class_id] for class_id in detections.class_id]
    
    # Get cached annotators
    box_annotator, label_annotator = get_annotators()
    
    # Prepare labels for visualization
    labels = [
        f"{class_name} {conf:.2f}"
        for class_name, conf in zip(detected_class_names, detections.confidence)
    ]
    
    # Annotate image
    annotated_image = pil_image.copy()
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, 
        detections=detections,
        labels=labels
    )
    annotated_image_np = np.asarray(annotated_image)
    
    # Optimized class counting
    classcounts = dict(Counter(detected_class_names))
    
    return annotated_image_np, classcounts


# --- Bagian Streamlit Utama ---

st.title("Skin Problem Detection")

# Model selection and settings
col_sel, col_conf = st.columns([2, 1])
with col_sel:
    selected_model = st.selectbox("Select Usecase", ["Skin Problem"])
with col_conf:
    conf_threshold = st.slider("Confidence", 0.0, 1.0, 0.25, 0.05)

# Map selection to model filename
model_map = {
    "Skin Problem": "skinproblem"
}

# Load model with caching (only loads once per model type)
with st.spinner(f"Loading {selected_model} model..."):
    model = load_model(model_map[selected_model])

st.success(f"✅ {selected_model} model loaded!")

# File upload
uploaded_file = st.file_uploader("Upload Image", accept_multiple_files=False, type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    # Detect button
    if st.button("🔍 Detect Objects", type="primary"):
        bytes_data = uploaded_file.getvalue()
        
        # Run detection with spinner
        with st.spinner("Detecting objects..."):
            annotated_image_rgb, classcounts = detector_pipeline_pillow(bytes_data, model, conf_threshold)
        
        # Display results
        st.subheader("Detection Results")
        st.image(annotated_image_rgb, caption="Detected Objects", use_container_width=True)
        
        # Display class counts in a nice format
        if classcounts:
            st.subheader("📊 Object Counts")
            col1, col2 = st.columns([1, 2])
            with col1:
                for class_name, count in classcounts.items():
                    st.metric(label=class_name, value=count)
        else:
            st.info("No objects detected in the image.")

        # Store detection results when detection happens
        st.session_state["last_detections"] = classcounts

    # --- LLM Assistant Section ---
    st.divider()
    st.header("AI Skin Advisor")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask about your skin condition or product recommendations..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get detection results context (if available from this run, checking if 'classcounts' exists locally)
        # Note: 'classcounts' is only defined inside the button click scope above.
        # We need to persist the detection results to session_state to use them in the chat.
        
        current_detections = st.session_state.get("last_detections", {})
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_skin_advice(current_detections, prompt, st.session_state.messages)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})