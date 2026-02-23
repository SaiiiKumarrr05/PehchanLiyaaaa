import torch
from PIL import Image
import evaluate

bleu = evaluate.load("bleu")


def generate_caption(model, image_path, device, max_length=20):

    image = Image.open(image_path).convert("RGB")

    generated = torch.tensor(
        [[model.tokenizer.eos_token_id]],
        device=device
    )

    for _ in range(max_length):

        outputs = model(image, generated, device)
        logits = outputs.logits[:, -1, :]

        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == model.tokenizer.eos_token_id:
            break

    caption = model.tokenizer.decode(
        generated[0],
        skip_special_tokens=True
    )

    return caption.strip()


def evaluate_model(model, data, dataset_path, device):

    predictions = []
    references = []

    for item in data[:10]:

        image_file = item["image"]
        true_caption = item["caption"]

        image_path = f"{dataset_path}/images/{image_file}"

        pred = generate_caption(model, image_path, device)

        predictions.append(pred)
        references.append([true_caption])

    score = bleu.compute(
        predictions=predictions,
        references=references
    )

    print("\nBLEU Score:", score)
