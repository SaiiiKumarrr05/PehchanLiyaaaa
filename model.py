import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class CaptionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ViT Encoder
        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        self.processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )

        # GPT2 Decoder
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2")

        # Fix padding
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Projection layer
        self.projection = nn.Linear(
            self.vit.config.hidden_size,
            self.decoder.config.hidden_size
        )

    def forward(self, image, generated_ids, device):

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            vit_output = self.vit(pixel_values=pixel_values)
            image_features = vit_output.last_hidden_state[:, 0, :]
            image_features = self.projection(image_features)

        image_embedding = image_features.unsqueeze(1)

        token_embeddings = self.decoder.transformer.wte(generated_ids)

        inputs_embeds = torch.cat(
            [image_embedding, token_embeddings],
            dim=1
        )

        outputs = self.decoder(inputs_embeds=inputs_embeds)

        return outputs
