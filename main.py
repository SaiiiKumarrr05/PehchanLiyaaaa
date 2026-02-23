"""
ViT Encoder + GPT2 Decoder
Training + Evaluation
BLEU + ROUGE Metrics
"""

import os
import json
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    ViTModel,
    ViTImageProcessor,
    GPT2Tokenizer,
    GPT2LMHeadModel
)
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer


# -------------------------------------------------
# CONFIG
# -------------------------------------------------

DATASET_PATH = "/Users/saadhanagroup/Downloads/multimodal_captioning/archive"
IMAGES_FOLDER = os.path.join(DATASET_PATH, "Images")
JSON_FILE     = os.path.join(DATASET_PATH, "captions.json")
MODEL_SAVE    = "models/my_captioning_model2.pt"

MAX_LENGTH   = 20       # increased from 15 for richer captions
EVAL_SAMPLES = 50
BEAM_WIDTH   = 3

# Training config
TRAIN        = True     # set False to skip training and just evaluate
EPOCHS       = 10
BATCH_SIZE   = 8
LR           = 1e-4
TRAIN_SPLIT  = 0.9      # 90% train, 10% val

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

with open(JSON_FILE, "r") as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")

split = int(len(data) * TRAIN_SPLIT)
train_data = data[:split]
val_data   = data[split:]
print(f"Train: {len(train_data)} | Val: {len(val_data)}")


# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------

print("Loading models...")

vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

decoder = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

projection = nn.Linear(
    vit.config.hidden_size,      # 768
    decoder.config.hidden_size   # 768
).to(device)

# Freeze ViT — it already knows how to see
for param in vit.parameters():
    param.requires_grad = False

print("ViT frozen. Training projection + GPT-2 decoder.")


# -------------------------------------------------
# FIND IMAGE FILE
# -------------------------------------------------

def find_image_file(item):
    if "image" in item:
        return item["image"]
    if "file_name" in item:
        return item["file_name"]
    if "image_id" in item:
        image_id = str(item["image_id"])
        for filename in os.listdir(IMAGES_FOLDER):
            if image_id in filename:
                return filename
    return None


# -------------------------------------------------
# DATASET CLASS
# -------------------------------------------------

class CaptionDataset(Dataset):
    def __init__(self, data, images_folder, processor, tokenizer, max_length=MAX_LENGTH):
        # Filter out items with missing images upfront
        self.samples = []
        for item in data:
            image_file = find_image_file(item)
            if image_file is None:
                continue
            image_path = os.path.join(images_folder, image_file)
            if not os.path.exists(image_path):
                continue
            caption = item.get("caption", "")
            if not caption:
                continue
            self.samples.append((image_path, caption))

        self.processor  = processor
        self.tokenizer  = tokenizer
        self.max_length = max_length
        print(f"  Dataset ready: {len(self.samples)} valid samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, caption = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        tokens = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = tokens.input_ids.squeeze(0)

        # Mask padding so loss ignores it
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return pixel_values, input_ids, labels


# -------------------------------------------------
# FEATURE EXTRACTION  (used in eval + inference)
# -------------------------------------------------

def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        vit_output = vit(pixel_values=pixel_values)
        image_features = vit_output.last_hidden_state[:, 0, :]
        image_features = projection(image_features)

    return image_features.unsqueeze(1)


# -------------------------------------------------
# GREEDY DECODING
# -------------------------------------------------

def generate_greedy(image_embedding):
    generated = torch.tensor([[tokenizer.eos_token_id]], device=device)

    for _ in range(MAX_LENGTH):
        with torch.no_grad():
            token_embeds   = decoder.transformer.wte(generated)
            inputs_embeds  = torch.cat([image_embedding, token_embeds], dim=1)
            outputs        = decoder(inputs_embeds=inputs_embeds)

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
        generated  = torch.cat([generated, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


# -------------------------------------------------
# BEAM SEARCH
# -------------------------------------------------

def generate_beam(image_embedding):
    beams = [(torch.tensor([[tokenizer.eos_token_id]], device=device), 0.0)]

    for _ in range(MAX_LENGTH):
        new_beams = []

        for seq, score in beams:
            with torch.no_grad():
                token_embeds  = decoder.transformer.wte(seq)
                inputs_embeds = torch.cat([image_embedding, token_embeds], dim=1)
                outputs       = decoder(inputs_embeds=inputs_embeds)

            probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
            topk  = torch.topk(probs, BEAM_WIDTH)

            for i in range(BEAM_WIDTH):
                next_token = topk.indices[0][i].unsqueeze(0).unsqueeze(0)
                new_seq    = torch.cat([seq, next_token], dim=1)
                new_score  = score + torch.log(topk.values[0][i])
                new_beams.append((new_seq, new_score))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:BEAM_WIDTH]

    return tokenizer.decode(beams[0][0][0], skip_special_tokens=True)


# -------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------

def train():
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)

    os.makedirs(os.path.dirname(MODEL_SAVE), exist_ok=True)

    train_dataset = CaptionDataset(train_data, IMAGES_FOLDER, processor, tokenizer)
    val_dataset   = CaptionDataset(val_data,   IMAGES_FOLDER, processor, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer  = AdamW(
        list(projection.parameters()) + list(decoder.parameters()),
        lr=LR, weight_decay=0.01
    )
    loss_fn    = nn.CrossEntropyLoss(ignore_index=-100)
    best_val   = float("inf")

    for epoch in range(EPOCHS):

        # ---- TRAIN ----
        projection.train()
        decoder.train()
        train_loss = 0.0

        for pixel_values, input_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            pixel_values = pixel_values.to(device)
            input_ids    = input_ids.to(device)
            labels       = labels.to(device)

            # Extract image features
            with torch.no_grad():
                vit_output = vit(pixel_values=pixel_values)
                image_feat = vit_output.last_hidden_state[:, 0, :]

            image_emb     = projection(image_feat).unsqueeze(1)          # (B, 1, 768)
            token_embeds  = decoder.transformer.wte(input_ids)           # (B, T, 768)
            inputs_embeds = torch.cat([image_emb, token_embeds], dim=1)  # (B, 1+T, 768)

            outputs = decoder(inputs_embeds=inputs_embeds)

            # Shift: predict next token
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, :].contiguous()

            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        # ---- VALIDATION ----
        projection.eval()
        decoder.eval()
        val_loss = 0.0

        with torch.no_grad():
            for pixel_values, input_ids, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                pixel_values = pixel_values.to(device)
                input_ids    = input_ids.to(device)
                labels       = labels.to(device)

                vit_output    = vit(pixel_values=pixel_values)
                image_feat    = vit_output.last_hidden_state[:, 0, :]
                image_emb     = projection(image_feat).unsqueeze(1)
                token_embeds  = decoder.transformer.wte(input_ids)
                inputs_embeds = torch.cat([image_emb, token_embeds], dim=1)

                outputs      = decoder(inputs_embeds=inputs_embeds)
                shift_logits = outputs.logits[:, :-1, :].contiguous()
                shift_labels = labels[:, :].contiguous()

                loss      = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        print(f"\nEpoch {epoch+1} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        # Save best model
        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "projection": projection.state_dict(),
                "decoder":    decoder.state_dict(),
            }, MODEL_SAVE)
            print(f"  ✅ Best model saved (val loss: {best_val:.4f})")

    print("\nTraining complete!")


# -------------------------------------------------
# LOAD SAVED WEIGHTS
# -------------------------------------------------

def load_trained_weights():
    if os.path.exists(MODEL_SAVE):
        print(f"Loading trained weights from {MODEL_SAVE}...")
        checkpoint = torch.load(MODEL_SAVE, map_location=device)
        projection.load_state_dict(checkpoint["projection"])
        decoder.load_state_dict(checkpoint["decoder"])
        print("Weights loaded.")
    else:
        print("No saved weights found — using untrained model.")


# -------------------------------------------------
# EVALUATION
# -------------------------------------------------

def evaluate():
    projection.eval()
    decoder.eval()

    bleu_greedy, bleu_beam   = [], []
    rouge_greedy, rouge_beam = [], []
    scorer_obj = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    print("\nEvaluating...")
    processed  = 0
    demo_shown = 0

    for item in tqdm(data):
        if processed >= EVAL_SAMPLES:
            break

        image_file = find_image_file(item)
        if image_file is None:
            continue

        image_path = os.path.join(IMAGES_FOLDER, image_file)
        if not os.path.exists(image_path):
            continue

        true_caption = item.get("caption", "")
        if not true_caption:
            continue

        image_embedding = extract_features(image_path)
        greedy_caption  = generate_greedy(image_embedding)
        beam_caption    = generate_beam(image_embedding)

        if demo_shown < 3:
            print("\n------------------------------")
            print(f"Image   : {image_file}")
            print(f"True    : {true_caption}")
            print(f"Greedy  : {greedy_caption}")
            print(f"Beam    : {beam_caption}")
            print("------------------------------")
            demo_shown += 1

        ref_tokens    = true_caption.lower().split()
        greedy_tokens = greedy_caption.lower().split()
        beam_tokens   = beam_caption.lower().split()

        bleu_greedy.append(sentence_bleu([ref_tokens], greedy_tokens))
        bleu_beam.append(sentence_bleu([ref_tokens], beam_tokens))

        rouge_greedy.append(scorer_obj.score(true_caption, greedy_caption)["rougeL"].fmeasure)
        rouge_beam.append(scorer_obj.score(true_caption, beam_caption)["rougeL"].fmeasure)

        processed += 1

    print("\n================ FINAL RESULTS ================")
    print(f"Greedy BLEU  : {sum(bleu_greedy)/len(bleu_greedy):.4f}")
    print(f"Beam   BLEU  : {sum(bleu_beam)/len(bleu_beam):.4f}")
    print(f"Greedy ROUGE : {sum(rouge_greedy)/len(rouge_greedy):.4f}")
    print(f"Beam   ROUGE : {sum(rouge_beam)/len(rouge_beam):.4f}")
    print("================================================")


# -------------------------------------------------
# MAIN
# -------------------------------------------------

if __name__ == "__main__":

    if TRAIN:
        train()

    load_trained_weights()
    evaluate()