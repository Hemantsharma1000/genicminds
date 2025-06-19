# File: src/train_caption.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration, Trainer, TrainingArguments
from PIL import Image
from tqdm import tqdm

from utils import DEVICE

# ── 1. Custom Dataset ─────────────────────────────────────────────────────────
class AnimalCaptionDataset(Dataset):
    def __init__(self, annotations_file: str, processor: BlipProcessor):
        """
        annotations_file: CSV with columns [image_path, caption, explicit_label]
        processor: BlipProcessor to tokenize text & process images
        """
        self.df = pd.read_csv(annotations_file)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        caption_text = row["caption"]

        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=image, text=caption_text, return_tensors="pt")

        item = {
            "pixel_values": inputs.pixel_values.squeeze(0),
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
        }
        return item

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0
    )
    return {
        "pixel_values": pixel_values.to(DEVICE),
        "input_ids": input_ids.to(DEVICE),
        "attention_mask": attention_mask.to(DEVICE),
    }

# ── 2. Main Training Function ─────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune BLIP on Animal Captions")
    parser.add_argument(
        "--train_csv", type=str, default="data/train_annotations.csv",
        help="Path to training CSV with [image_path,caption,explicit_label]"
    )
    parser.add_argument(
        "--val_csv", type=str, default="data/val_annotations.csv",
        help="Path to validation CSV"
    )
    parser.add_argument(
        "--output_dir", type=str, default="blip_animal_interactions",
        help="Where to save the fine-tuned BLIP model"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    # 1) Load pretrained BLIP large + processor
    print("Loading pretrained BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(DEVICE)

    # 2) Create Datasets + DataLoaders
    train_ds = AnimalCaptionDataset(args.train_csv, processor)
    val_ds = AnimalCaptionDataset(args.val_csv, processor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 3) Setup Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda batch: collate_fn(batch),
    )

    # 4) Train
    print("Starting fine-tuning BLIP...")
    trainer.train()
    print("Saving fine-tuned model to", args.output_dir)
    trainer.save_model(args.output_dir)
