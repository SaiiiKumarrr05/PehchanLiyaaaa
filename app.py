import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import base64
from io import BytesIO

# -------------------------------
# CONFIG
# -------------------------------
MODEL_NAME = "models/my_captioning_model2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="Pehchan कौन ¿",
    page_icon="think.jpg",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# LOAD THINK IMAGE FOR TITLE
# -------------------------------

def img_to_base64(img_path):
    try:
        img = Image.open(img_path).resize((48, 48))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    except:
        return None

def gif_to_base64(gif_path):
    try:
        with open(gif_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

think_b64  = img_to_base64("think.jpg")
gotcha_b64 = gif_to_base64("gotcha.gif")

think_html = (
    f'<img src="data:image/png;base64,{think_b64}" '
    f'style="width:52px;height:52px;border-radius:50%;object-fit:cover;'
    f'vertical-align:middle;margin-right:12px;box-shadow:0 2px 8px rgba(0,0,0,0.25)"/>'
    if think_b64 else "🧠"
)

gotcha_html = (
    f'<img src="data:image/gif;base64,{gotcha_b64}" '
    f'style="width:220px;border-radius:12px;margin-top:12px;"/>'
    if gotcha_b64 else "🎯 Pehchanliyaaaaaa!"
)

# -------------------------------
# STYLING — DARK + LIGHT MODE
# -------------------------------

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ---- ROOT VARIABLES ---- */
:root {{
    --accent: #00e5b0;
    --accent2: #ff6b6b;
    --radius: 14px;
    --font-main: 'Syne', sans-serif;
    --font-mono: 'DM Mono', monospace;
}}

/* ---- GLOBAL ---- */
html, body, [class*="css"] {{
    font-family: var(--font-main) !important;
}}

/* ---- LIGHT MODE ---- */
@media (prefers-color-scheme: light) {{
    :root {{
        --bg: #f4f4f0;
        --surface: #ffffff;
        --surface2: #ebebeb;
        --text-primary: #111111;
        --text-secondary: #555555;
        --border: rgba(0,0,0,0.08);
        --shadow: 0 4px 24px rgba(0,0,0,0.08);
    }}
}}

/* ---- DARK MODE ---- */
@media (prefers-color-scheme: dark) {{
    :root {{
        --bg: #0e0f11;
        --surface: #1a1c20;
        --surface2: #242629;
        --text-primary: #f0f0f0;
        --text-secondary: #888888;
        --border: rgba(255,255,255,0.07);
        --shadow: 0 4px 24px rgba(0,0,0,0.4);
    }}
}}

/* ---- STREAMLIT OVERRIDES ---- */
.stApp {{
    background-color: var(--bg) !important;
}}

section[data-testid="stSidebar"] {{
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}}

/* ---- HEADER BLOCK ---- */
.hero-header {{
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 32px 0 8px 0;
}}

.hero-title {{
    font-size: 2.8rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    line-height: 1;
    margin: 0;
}}

.hero-title span {{
    color: var(--accent);
}}

.hero-sub {{
    font-family: var(--font-mono);
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin: 6px 0 32px 0;
    letter-spacing: 0.04em;
}}

/* ---- DEVICE BADGE ---- */
.device-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 5px 14px;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-bottom: 28px;
}}

.device-badge .dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 6px var(--accent);
    animation: pulse 2s infinite;
}}

@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.4; }}
}}

/* ---- UPLOAD AREA ---- */
[data-testid="stFileUploader"] {{
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 12px !important;
    transition: border-color 0.2s;
}}

[data-testid="stFileUploader"]:hover {{
    border-color: var(--accent) !important;
}}

/* ---- IMAGE DISPLAY ---- */
[data-testid="stImage"] img {{
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow) !important;
}}

/* ---- CAPTION BOX ---- */
.caption-box {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: var(--radius);
    padding: 20px 24px;
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 16px 0;
    box-shadow: var(--shadow);
    letter-spacing: -0.01em;
}}

.caption-label {{
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--accent);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
}}

/* ---- SUCCESS TOAST ---- */
[data-testid="stAlert"] {{
    background: var(--surface) !important;
    border: 1px solid var(--accent) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
}}

/* ---- BUTTON ---- */
.stButton > button {{
    background: var(--accent) !important;
    color: #000000 !important;
    font-family: var(--font-main) !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 14px 28px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 0 0 rgba(0,229,176,0.4) !important;
}}

.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0,229,176,0.35) !important;
}}

.stButton > button:active {{
    transform: translateY(0px) !important;
}}

/* ---- SPINNER ---- */
[data-testid="stSpinner"] {{
    color: var(--accent) !important;
}}

/* ---- SECTION HEADER ---- */
.section-tag {{
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--accent);
    background: rgba(0,229,176,0.1);
    border: 1px solid rgba(0,229,176,0.25);
    border-radius: 6px;
    padding: 3px 10px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}}

/* ---- DIVIDER ---- */
hr {{
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 24px 0 !important;
}}

/* ---- ATTENTION HEADER ---- */
h4 {{
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}}

/* ---- WARNING ---- */
[data-testid="stWarning"] {{
    background: rgba(255,107,107,0.08) !important;
    border: 1px solid rgba(255,107,107,0.3) !important;
    border-radius: var(--radius) !important;
}}

/* ---- COLUMNS GAP ---- */
[data-testid="column"] {{
    padding: 0 12px !important;
}}

/* ---- HIDE STREAMLIT BRANDING ---- */
#MainMenu, footer, header {{
    visibility: hidden;
}}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------

st.markdown(f"""
<div class="hero-header">
    {think_html}
    <h1 class="hero-title">Pehchan <span>कौन</span> ¿</h1>
</div>
<p class="hero-sub">// ViT Encoder + GPT-2 Decoder &nbsp;·&nbsp; Multimodal NLP</p>
<div class="device-badge">
    <span class="dot"></span>
    Running on {DEVICE.upper()}
</div>
""", unsafe_allow_html=True)

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

@st.cache_resource
def load_model():
    with st.spinner("Warming up the model..."):
        model = VisionEncoderDecoderModel.from_pretrained(
            MODEL_NAME, attn_implementation="eager"
        ).to(DEVICE)
        model.config.output_attentions = True
        processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id
    return model, processor, tokenizer

def get_smooth_mask(cross_attentions, token_idx, target_size):
    try:
        # cross_attentions is a tuple of steps, each step is a tuple of layers
        step_attn = cross_attentions[token_idx]  # attention at this generation step

        # Handle different shapes depending on model output
        # step_attn is tuple of layers → take last layer
        last_layer = step_attn[-1]  # (batch, heads, 1, seq_len)

        # Squeeze out batch + query dims safely
        attn = last_layer.squeeze(0).mean(dim=0).squeeze(0)  # → (seq_len,)

        # Remove CLS token if present
        if attn.shape[0] == 197:
            attn = attn[1:]  # → (196,)

        mask = attn.reshape(14, 14).detach().cpu().float().numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        return mask_img.resize(target_size, resample=Image.BICUBIC)

    except Exception as e:
        # Return blank mask if anything goes wrong
        return Image.fromarray(np.zeros((target_size[1], target_size[0]), dtype=np.uint8))

model, processor, tokenizer = load_model()

# -------------------------------
# APP LOGIC
# -------------------------------

st.markdown('<div class="section-tag">📂 Upload</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drop an image to identify", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-tag">🖼 Input Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with col2:
        st.markdown('<div class="section-tag">⚡ Generate</div>', unsafe_allow_html=True)

        if st.button("Generate Caption & Visualize", use_container_width=True):
            with st.spinner("Analyzing pixels and attention..."):
                pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)

                outputs = model.generate(
                    pixel_values,
                    max_length=15,
                    num_beams=1,
                    output_attentions=True,
                    return_dict_in_generate=True
                )

                gen_ids = outputs.sequences[0]
                caption = tokenizer.decode(gen_ids, skip_special_tokens=True)

                tokens_raw = tokenizer.convert_ids_to_tokens(gen_ids)
                valid_indices = [i for i, t in enumerate(tokens_raw)
                                 if t not in tokenizer.all_special_tokens and t.strip() not in ['', 'Ġ']]

            st.markdown(f"""
            <div class="caption-box">
                <div class="caption-label">Generated Caption</div>
                📝 {caption}
            </div>
            """, unsafe_allow_html=True)

            # ---- GOTCHA GIF ----
            st.markdown(gotcha_html, unsafe_allow_html=True)
            st.success("Pehchanliyaaaaaaaa")
            # --- ATTENTION PLOT ---
            if outputs.cross_attentions:
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("#### 🧠 Where the model looked")

                cross_attentions = outputs.cross_attentions
                all_masks = [np.array(get_smooth_mask(cross_attentions, i, image.size)) for i in valid_indices]

                if all_masks:
                    combined_mask = np.mean(all_masks, axis=0)

                    fig, ax = plt.subplots(figsize=(8, 8))
                    fig.patch.set_alpha(0)
                    ax.set_facecolor('none')
                    ax.imshow(image)
                    ax.imshow(combined_mask, cmap='inferno', alpha=0.55)
                    ax.axis('off')
                    plt.tight_layout(pad=0)

                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.warning("Could not generate attention map for these tokens.")
