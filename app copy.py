"""
FINAL PROFESSOR SAFE VERSION
Accurate Image Captioning
Optimized + Clean UI
"""

import streamlit as st
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# -------------------------------
# CONFIG
# -------------------------------

MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="AI Image Captioning",
    page_icon="📸",
    layout="wide"
)

# -------------------------------
# STYLING
# -------------------------------

st.markdown("""
<style>
body {
    background-color: #f0f2f6;
}
.big-title {
    font-size: 42px;
    font-weight: 700;
    color: #1f2937;
}
.subtitle {
    font-size: 18px;
    color: #4b5563;
}
.caption-box {
    padding: 20px;
    background-color: white;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    font-size: 22px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>📸 Pehchan कौन ¿</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>ViT Encoder + GPT2 Decoder </div>", unsafe_allow_html=True)

st.sidebar.info(f"Running on: **{DEVICE.upper()}**")

# -------------------------------
# LOAD MODEL (cached)
# -------------------------------

@st.cache_resource
def load_model():
    with st.spinner("Loading AI model (first time may take 30-60 seconds)..."):
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(DEVICE)
        processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, processor, tokenizer

model, processor, tokenizer = load_model()

# -------------------------------
# IMAGE UPLOAD
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:

        if st.button("Generate Caption", use_container_width=True):

            with st.spinner("Analyzing image..."):

                pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)

                # SAFE GENERATION SETTINGS
                output_ids = model.generate(
                    pixel_values,
                    max_length=12,
                    num_beams=3,              # beam search for accuracy
                    early_stopping=True,
                    no_repeat_ngram_size=2,   # prevents repetition
                    do_sample=False           # disables randomness
                )

                caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            st.markdown(
                f"<div class='caption-box'>📝 {caption}</div>",
                unsafe_allow_html=True
            )

            st.success("Pehchanliyaaaaaa")