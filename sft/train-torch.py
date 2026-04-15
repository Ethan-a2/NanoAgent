import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from utils.tokenizer import get_tokenizer


@dataclass
class TrainConfig:
    MODEL = "HuggingFaceTB/SmolLM2-135M"
    EPOCHS = 1
    BATCH_SIZE = 1
    CONTEXT_LEN = 2048
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.1
    WARMUP_RATIO = 0.05
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_GRAD_NORM = 1.0
    SAVE_PATH = "weights/SmolLM2-135M-torch-sft"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    config = TrainConfig()
    print(
        json.dumps(
            {
                k: v
                for k, v in config.__dict__.items()
                if not k.startswith("__") and not callable(v)
            },
            indent=2,
        )
    )

    print(f"Using device: {config.DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL,
        torch_dtype=torch.bfloat16 if config.DEVICE == "cuda" else torch.float32,
    ).to(config.DEVICE)

    print(
        f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters"
    )

    train_ds = load_dataset(
        "json",
        data_files="data/datasets/Smollm2_base_train_2048_nemotron_instruct_fc_base.jsonl",
        split="train",
    )

    test_ds = load_dataset(
        "json",
        data_files="data/datasets/Smollm2_base_test_2048_nemotron_instruct_fc_base.jsonl",
        split="train",
    )

    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    class SFTDataset(TorchDataset):
        def __init__(self, dataset, tokenizer, config):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.config = config

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            messages = item["messages"]

            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            enc = self.tokenizer(
                text,
                max_length=self.config.CONTEXT_LEN,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = enc["input_ids"].squeeze(0)
            labels = input_ids.clone()

            return {"input_ids": input_ids, "labels": labels}

    train_dataset = SFTDataset(train_ds, tokenizer, config)
    test_dataset = SFTDataset(test_ds, tokenizer, config)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    total_steps = (
        len(train_loader) * config.EPOCHS // config.GRADIENT_ACCUMULATION_STEPS
    )
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()
    global_step = 0

    for epoch in range(config.EPOCHS):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            input_ids = batch["input_ids"].to(config.DEVICE)
            labels = batch["labels"].to(config.DEVICE)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / config.GRADIENT_ACCUMULATION_STEPS

            loss.backward()

            if (global_step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if global_step % 100 == 0:
                print(
                    f"Step {global_step}, Loss: {loss.item() * config.GRADIENT_ACCUMULATION_STEPS:.4f}"
                )

        eval_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(config.DEVICE)
                labels = batch["labels"].to(config.DEVICE)
                outputs = model(input_ids=input_ids, labels=labels)
                eval_loss += outputs.loss.item()
        eval_loss /= len(test_loader)
        print(f"Epoch {epoch + 1}, Eval Loss: {eval_loss:.4f}")
        model.train()

        Path(config.SAVE_PATH).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(config.SAVE_PATH)
        tokenizer.save_pretrained(config.SAVE_PATH)
        print(f"Model saved to {config.SAVE_PATH}")


if __name__ == "__main__":
    main()
