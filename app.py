import os
import csv
import time
from datetime import datetime
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
st.set_page_config(
    page_title="FOODVISION-AI | FOOD FRESHNESS PREDICTION",
    page_icon="🥗",
    layout="wide"
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_mobilenetv2_food_model.keras"
)
TRAIN_DIR = os.path.join(
    BASE_DIR,
    "prepared_dataset",
    "train"
)
FEEDBACK_DIR = os.path.join(BASE_DIR, "feedback")
FEEDBACK_FILE = os.path.join(
    FEEDBACK_DIR,
    "user_feedback.csv"
)
IMAGE_SIZE = (224, 224)
st.markdown("""
<style>
    .stApp {
        background-color: #090D16;
        color: #F8FAFC;
    }
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 0.75rem 1.25rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .live-badge {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid #10B981;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .offline-badge {
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid #EF4444;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .ai-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .badge-fresh {
        background-color: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        color: #34D399;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        text-align: center;
        font-size: 1.1rem;
    }
    .badge-rotten {
        background-color: rgba(239, 68, 68, 0.15);
        border: 1px solid #EF4444;
        color: #F87171;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        text-align: center;
        font-size: 1.1rem;
    }
    .meta-box {
        background: rgba(30, 41, 59, 0.4);
        border-left: 3px solid #38BDF8;
        padding: 0.75rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #94A3B8;
        font-family: monospace;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)
if not os.path.isdir(TRAIN_DIR):
    st.error(
        f"Training dataset directory not found: {TRAIN_DIR}"
    )
    st.info(
        "Please ensure that prepared_dataset/train exists."
    )
    st.stop()
class_names = sorted([
    folder
    for folder in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, folder))
])
if len(class_names) == 0:
    st.error("No food classes found in the training directory.")
    st.stop()
@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}"
        )
    model = tf.keras.models.load_model(MODEL_PATH)
    return model
model_available = os.path.isfile(MODEL_PATH)
if model_available:
    status_html = '<span class="live-badge">🟢 MODEL FILE AVAILABLE</span>'
else:
    status_html = '<span class="offline-badge">🔴 MODEL FILE MISSING</span>'
st.markdown(f'<div class="top-bar"><div><span style="font-weight:700;font-size:1.2rem;color:#38BDF8;">🥗 FOODVISION-AI</span><span style="color:#64748B;margin-left:10px;font-size:0.85rem;">| Food Freshness Prediction AI |</span></div><div>{status_html}<span style="color:#94A3B8;margin-left:8px;font-size:0.75rem;">{len(class_names)} CLASSES</span></div></div>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["⚡ FOOD SAMPLES", "📈 Model Telemetry", "📂 Dataset Registry", "💬 User Feedback"])
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown(
            '<div class="ai-card">',
            unsafe_allow_html=True
        )
        st.subheader("📥 INPUT IMAGE OF THE FOOD")
        uploaded_file = st.file_uploader(
            "Upload food item image for inspection",
            type=["jpg", "jpeg", "png", "webp"]
        )
        image = None
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(
                    image,
                    caption=f"Source: {uploaded_file.name}",
                    use_container_width=True
                )
                st.markdown(f"""
                <div class="meta-box">
                    <b>Dimensions:</b>
                    {image.size[0]} x {image.size[1]} px
                    <br>
                    <b>Color Space:</b> RGB
                    <br>
                    <b>Target Resolution:</b>
                    224 x 224 px (Auto-scaled)
                </div>
                """, unsafe_allow_html=True)
            except (UnidentifiedImageError, OSError, ValueError):
                st.error(
                    "Invalid image. Please upload a valid food image."
                )
                image = None
        else:
            st.info(
                "Upload a high-resolution photo of fruit, "
                "vegetable, meat, fish, dairy, or other food."
            )
        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            '<div class="ai-card">',
            unsafe_allow_html=True
        )
        st.subheader("🔬 AI Freshness Result")
        if image is not None:
            if st.button(
                "Execute Deep Analysis",
                type="primary",
                use_container_width=True
            ):
                start_time = time.time()
                with st.spinner(
                    "Processing image through MobileNetV2..."
                ):
                    try:
                        if not model_available:
                            raise FileNotFoundError(
                                f"Model not found: {MODEL_PATH}"
                            )
                        model = load_model()
                        resized_image = image.resize(IMAGE_SIZE)
                        image_array = np.array(
                            resized_image,
                            dtype=np.float32
                        )
                        image_array = np.expand_dims(
                            image_array,
                            axis=0
                        )
                        predictions = model.predict(
                            image_array,
                            verbose=0
                        )
                        inference_time = time.time() - start_time
                        if predictions.ndim != 2:
                            raise ValueError(
                                "Unexpected model output dimensions."
                            )
                        if predictions.shape[1] != len(class_names):
                            raise ValueError(
                                "Class dimension mismatch! "
                                f"Model has {predictions.shape[1]} "
                                f"outputs, but dataset has "
                                f"{len(class_names)} classes."
                            )
                        if not np.all(np.isfinite(predictions)):
                            raise ValueError(
                                "Model returned invalid prediction values."
                            )
                        predicted_index = np.argmax(
                            predictions[0]
                        )
                        predicted_class = class_names[
                            predicted_index
                        ]
                        confidence = float(
                            predictions[0][predicted_index] * 100
                        )
                        if predicted_class.endswith("_Fresh"):
                            food_name = predicted_class[:-6]
                            freshness = "Fresh"
                        elif predicted_class.endswith("_Rotten"):
                            food_name = predicted_class[:-7]
                            freshness = "Rotten"
                        else:
                            food_name = predicted_class
                            freshness = "Unknown"
                        if freshness == "Fresh":
                            st.markdown(
                                '<div class="badge-fresh">'
                                '✅ VERIFIED FRESH'
                                '</div>',
                                unsafe_allow_html=True
                            )
                        elif freshness == "Rotten":
                            st.markdown(
                                '<div class="badge-rotten">'
                                '⚠️ SPOILED / ROTTEN DETECTED'
                                '</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.warning(
                                "Freshness status ambiguous."
                            )
                        st.markdown(
                            f"""
                            <h3 style="
                                margin-top: 1rem;
                                margin-bottom: 0px;
                            ">
                                Food: {food_name.replace('_', ' ')}
                            </h3>
                            """,
                            unsafe_allow_html=True
                        )
                        st.write(
                            f"Freshness: **{freshness}**"
                        )
                        st.write(
                            f"Confidence Score: **{confidence:.2f}%**"
                        )
                        st.progress(
                            max(0, min(100, int(confidence)))
                        )
                        st.markdown("---")
                        st.markdown(
                            "#### 📊 Top 5 Probabilities"
                        )
                        top_indices = np.argsort(
                            predictions[0]
                        )[::-1][:5]
                        for idx in top_indices:
                            cls_label = class_names[idx].replace(
                                "_", " "
                            )
                            score = float(
                                predictions[0][idx] * 100
                            )
                            st.write(
                                f"• `{cls_label}` — {score:.2f}%"
                            )
                        st.markdown("---")
                        st.markdown(
                            "#### ⚡ Inference Performance"
                        )
                        metric1, metric2 = st.columns(2)
                        metric1.metric(
                            "Inference Time",
                            f"{inference_time:.3f} sec"
                        )
                        metric2.metric(
                            "Image Resolution",
                            "224 × 224"
                        )
                        st.caption(
                            "Prediction is generated using "
                            "the trained MobileNetV2 model."
                        )
                    except Exception as error:
                        st.error(
                            f"Inference pipeline failed: {error}"
                        )
                        st.info(
                            "Check the model file, dataset class "
                            "directories, and model configuration."
                        )
            else:
                st.write(
                    "Click **Execute Deep Analysis** "
                    "to check the freshness of the food."
                )
        else:
            st.write(
                "Awaiting food image upload..."
            )
        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )
with tab2:
    st.markdown(
        '<div class="ai-card">',
        unsafe_allow_html=True
    )
    st.subheader("📈 Model Performance & Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Validation Accuracy",
        "94.05%"
    )
    c2.metric(
        "Loss Function",
        "Categorical Crossentropy"
    )
    c3.metric(
        "Optimizer",
        "Adam (lr=1e-4)"
    )
    c4.metric(
        "Active Classes",
        len(class_names)
    )
    st.markdown("---")
    st.markdown("### Pipeline Specifications")
    st.markdown("""
    - **Backbone Network:** MobileNetV2 pretrained on ImageNet.
    - **Model Architecture:** Convolutional Neural Network.
    - **Input Resolution:** 224 × 224 pixels.
    - **Input Normalization:** Uses the preprocessing
      configured during model training.
    - **Optimizer:** Adam.
    - **Loss Function:** Categorical Crossentropy.
    - **Dataset Partitioning:** Stratified split into
      70% Training, 15% Validation, and 15% Testing.
    """)
    st.markdown("---")
    st.markdown("### Model File Information")
    col_a, col_b = st.columns(2)
    col_a.write(
        "**Model Filename:**"
    )
    col_a.code(
        "best_mobilenetv2_food_model.keras"
    )
    col_b.write(
        "**Model File Status:**"
    )
    if model_available:
        col_b.success(
            "Model file found"
        )
        model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        col_b.write(
            f"File Size: {model_size:.2f} MB"
        )
    else:
        col_b.error(
            "Model file missing"
        )
    st.markdown('</div>', unsafe_allow_html=True)
with tab3:
    st.markdown(
        '<div class="ai-card">',
        unsafe_allow_html=True
    )
    st.subheader(
        f"📂 Class Directory ({len(class_names)} Categories)"
    )
    st.write(
        "Food categories available in the training dataset."
    )
    cols = st.columns(3)
    for i, name in enumerate(class_names, start=1):
        cols[(i - 1) % 3].write(
            f"`{i:02d}`. {name.replace('_', ' ')}"
        )
    st.markdown("---")
    st.markdown("### Dataset Information")
    st.write(
        f"Total food classes: **{len(class_names)}**"
    )
    st.write(
        f"Training directory: `{TRAIN_DIR}`"
    )
    st.markdown('</div>', unsafe_allow_html=True)
with tab4:
    st.markdown(
        '<div class="ai-card">',
        unsafe_allow_html=True
    )
    st.subheader(
        "💬 User Feedback & Evaluation"
    )
    st.write(
        "Help us improve the FOODVISION-AI model by sharing "
        "your experience or reporting misclassifications."
    )
    with st.form("feedback_form"):
        feedback_rating = st.radio(
            "How would you rate the prediction accuracy?",
            [
                "⭐ Excellent",
                "⭐ Good",
                "⭐ Needs Improvement"
            ],
            horizontal=True
        )
        feedback_text = st.text_area(
            "Comments, suggestions, or correction notes:",
            placeholder=(
                "E.g., The tomato was marked fresh, "
                "but it had minor spots..."
            )
        )
        submit_feedback = st.form_submit_button(
            "Submit Feedback",
            type="primary"
        )
        if submit_feedback:
            try:
                os.makedirs(
                    FEEDBACK_DIR,
                    exist_ok=True
                )
                file_exists = os.path.isfile(
                    FEEDBACK_FILE
                )
                with open(
                    FEEDBACK_FILE,
                    mode="a",
                    newline="",
                    encoding="utf-8"
                ) as file:
                    writer = csv.writer(file)
                    if not file_exists:
                        writer.writerow([
                            "Timestamp",
                            "Rating",
                            "Comments"
                        ])
                    writer.writerow([
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        feedback_rating,
                        feedback_text
                    ])
                st.success(
                    "Thank you! Your feedback has "
                    "been saved successfully."
                )
            except OSError as error:
                st.error(
                    f"Unable to save feedback: {error}"
                )
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown(
    "<br>",
    unsafe_allow_html=True
)
st.caption(
    "FOODVISION-AI Deep Learning Infrastructure "
    "• For research and demonstration purposes only."
)