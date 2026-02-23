# Pehchan कौन ¿ 🧠
### AI Image Captioning — ViT Encoder + GPT-2 Decoder

A multimodal deep learning project that generates natural language captions from images using a Vision Transformer (ViT) encoder and GPT-2 decoder, with cross-attention visualization.

---

## Demo

Upload any image → get an AI-generated caption + attention heatmap showing where the model looked.

---

## Architecture

```
Image → ViT Encoder → CLS Embedding → Projection Layer → GPT-2 Decoder → Caption
```

| Component | Model | Role |
|---|---|---|
| Encoder | `google/vit-base-patch16-224-in21k` | Extracts visual features from 16×16 image patches |
| Projection | `nn.Linear(768 → 768)` | Bridges ViT embedding space to GPT-2 input space |
| Decoder | `GPT-2` | Autoregressively generates caption tokens |

---

## Features

- **Greedy Decoding** — fast token-by-token generation
- **Beam Search** (width=3) — higher quality captions
- **Cross-Attention Heatmaps** — visualizes which image regions influenced each word
- **BLEU & ROUGE-L Evaluation** — quantitative benchmarking
- **Streamlit Web App** — interactive UI with dark/light mode

---

## Project Structure

```
multimodal_captioning/
├── app.py              # Streamlit web app
├── model.py            # CaptionModel class (ViT + Projection + GPT-2)
├── main.py             # Training loop + evaluation
├── utils.py            # Caption generation + BLEU evaluation helpers
├── think.jpg           # Title image asset
├── gotcha.gif          # Success gif
├── requirements.txt    # Dependencies
├── models/
│   └── my_captioning_model2/   # Saved fine-tuned model weights
└── archive/
    ├── Images/                 # Dataset images
    └── captions.json           # Ground truth captions
```

---

## Setup

**1. Clone / download the project**
```bash
cd multimodal_captioning
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your dataset**

Place your images in `archive/Images/` and captions in `archive/captions.json`.

Expected JSON format:
```json
[
  {"image": "img1.jpg", "caption": "A dog running on the beach"},
  {"image": "img2.jpg", "caption": "A cat sitting on a chair"}
]
```

---

## Training

```bash
python main.py
```

Set `TRAIN = True` in `main.py` to train. The model will:
- Freeze ViT weights (pretrained, no need to retrain)
- Train the projection layer + GPT-2 decoder
- Save the best checkpoint to `models/my_captioning_model2.pt`
- Evaluate on held-out samples and print BLEU + ROUGE scores

Key training config in `main.py`:
```python
EPOCHS      = 10
BATCH_SIZE  = 8
LR          = 1e-4
TRAIN_SPLIT = 0.9   # 90% train, 10% val
MAX_LENGTH  = 20
```

---

## Running the App

```bash
streamlit run app.py
```

Make sure `think.jpg` and `gotcha.gif` are in the same folder as `app.py`.

---

## Evaluation Metrics

| Metric | What it measures |
|---|---|
| BLEU | N-gram overlap between generated and reference caption |
| ROUGE-L | Longest Common Subsequence between generated and reference |

Run evaluation only (no training):
```python
# In main.py, set:
TRAIN = False
```

---

## Requirements

```
torch
torchvision
transformers
streamlit
Pillow
numpy
matplotlib
nltk
rouge-score
tqdm
evaluate
```

---

## Notes

- ViT is frozen during training — only the projection layer and GPT-2 are updated
- `num_beams=1` is used in the app for attention visualization (beam search breaks cross-attention mapping)
- Model is saved in HuggingFace format so `from_pretrained()` works directly in `app.py`

---

## Built With

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Streamlit](https://streamlit.io)
- [PyTorch](https://pytorch.org)
