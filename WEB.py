"""
STREAMLIT IMAGE CAPTIONING APP
ViT Encoder + GPT2 Decoder
Upload image → Generate caption
"""

import streamlit as st
import torch
from PIL import Image
from transformers import (
    ViTModel,
    ViTImageProcessor,
    GPT2Tokenizer,
    GPT2LMHeadModel
)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 15

st.set_page_config(page_title="Image Captioning", layout="centered")
st.title("📸 ViT + GPT2 Image Captioning")
st.sidebar.info(f"Running on: **{DEVICE.upper()}**")

# -------------------------------------------------
# LOAD MODELS (Cached)
# -------------------------------------------------

@st.cache_resource
def load_models():

    vit = ViTModel.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    ).to(DEVICE)

    processor = ViTImageProcessor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    decoder = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)

    tokenizer.pad_token = tokenizer.eos_token

    projection = torch.nn.Linear(
        vit.config.hidden_size,
        decoder.config.hidden_size
    ).to(DEVICE)

    vit.eval()
    decoder.eval()

    return vit, processor, tokenizer, decoder, projection


vit, processor, tokenizer, decoder, projection = load_models()

# -------------------------------------------------
# FEATURE EXTRACTION
# -------------------------------------------------

def extract_features(image):

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.no_grad():
        vit_output = vit(pixel_values=pixel_values)
        image_features = vit_output.last_hidden_state[:, 0, :]
        image_features = projection(image_features)

    return image_features.unsqueeze(1)


# -------------------------------------------------
# GREEDY DECODING
# -------------------------------------------------

def generate_caption(image_embedding):

    generated = torch.tensor([[tokenizer.eos_token_id]], device=DEVICE)

    for _ in range(MAX_LENGTH):

        with torch.no_grad():
            token_embeds = decoder.transformer.wte(generated)
            inputs_embeds = torch.cat([image_embedding, token_embeds], dim=1)
            outputs = decoder(inputs_embeds=inputs_embeds)

        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
        generated = torch.cat([generated, next_token], dim=1)

    caption = tokenizer.decode(generated[0], skip_special_tokens=True)
    return caption


# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Caption"):

        with st.spinner("Generating caption..."):

            image_embedding = extract_features(image)
            caption = generate_caption(image_embedding)

        st.success(f"📝 Caption: {caption}")