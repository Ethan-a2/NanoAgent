import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"

import random
from collections import defaultdict
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np


@dataclass
class TrainConfig:
    MODEL = "weights/SmolLM2-135M-torch-sft"
    EPOCHS = 1
    BATCH_SIZE = 1
    GEN_LEN = 384
    GROUP_SIZE = 8
    ITERS = 2000
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    EPSILON = 0.2
    TEMPERATURE = 0.4
    TOP_P = 0.9
    MAX_GRAD_NORM = 1.0
    SAVE_PATH = "weights/SmolLM2-135M-torch-grpo"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def get_logits(model, input_ids, attention_mask=None):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits


def generate(model, input_ids, max_new_tokens, temperature, top_p):
    model.eval()
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        logits = get_logits(model, generated)
        next_token_logits = logits[:, -1, :] / temperature

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                next_token_logits, descending=True
            )
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float("-inf")

        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=-1)

        if next_token.item() == 2:
            break

    return generated


def compute_grpo_loss(logits, old_logits, rewards, epsilon=0.2):
    log_probs = F.log_softmax(logits, dim=-1)
    old_log_probs = F.log_softmax(old_logits, dim=-1)

    ratio = torch.exp(log_probs - old_log_probs)
    ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

    loss = -ratio * rewards
    return loss.mean()


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

    model_old = AutoModelForCausalLM.from_pretrained(
        config.MODEL,
        torch_dtype=torch.bfloat16 if config.DEVICE == "cuda" else torch.float32,
    ).to(config.DEVICE)
    model_old.eval()

    print(
        f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters"
    )

    train_ds = load_dataset(
        "json",
        data_files="data/datasets/grpo_cache.pickle",
        split="train",
    )

    print(f"Train samples: {len(train_ds)}")

    optimizer = AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    model.train()
    global_step = 0
    all_rewards = []
    all_losses = []

    for iteration in tqdm(range(config.ITERS), desc="GRPO Training"):
        batch = random.choice(train_ds)
        prompt = batch["prompt"]
        scorer = batch["scorer"]

        input_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(config.DEVICE)

        rewards = []
        samples = []

        for _ in range(config.GROUP_SIZE):
            with torch.no_grad():
                generated = generate(
                    model_old,
                    input_ids,
                    config.GEN_LEN,
                    config.TEMPERATURE,
                    config.TOP_P,
                )
                generated_ids = generated[0][input_ids.shape[0] :]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                reward = scorer(response, False)
                rewards.append(reward)
                samples.append(generated)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=config.DEVICE)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        generated = generate(
            model, input_ids, config.GEN_LEN, config.TEMPERATURE, config.TEP_P
        )

        old_logits = get_logits(model_old, generated)
        new_logits = get_logits(model, generated)

        loss = compute_grpo_loss(new_logits, old_logits, rewards, config.EPSILON)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()

        all_rewards.append(rewards.mean().item())
        all_losses.append(loss.item())

        if (iteration + 1) % 100 == 0:
            print(
                f"Iter {iteration + 1}, Loss: {np.mean(all_losses[-100:]):.4f}, Reward: {np.mean(all_rewards[-100:]):.4f}"
            )

        if (iteration + 1) % 500 == 0:
            Path(config.SAVE_PATH).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(config.SAVE_PATH)
            tokenizer.save_pretrained(config.SAVE_PATH)
            print(f"Model saved to {config.SAVE_PATH}")

    Path(config.SAVE_PATH).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.SAVE_PATH)
    tokenizer.save_pretrained(config.SAVE_PATH)
    print(f"Training complete. Model saved to {config.SAVE_PATH}")


if __name__ == "__main__":
    main()
