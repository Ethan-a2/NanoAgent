import json
import os

import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TrainConfig:
    MODEL = "weights/SmolLM2-135M-torch-sft"  # SFT 训练后的模型
    GEN_LEN = 128
    GROUP_SIZE = 4
    ITERS = 5
    LEARNING_RATE = 1e-5
    EPSILON = 0.2
    TEMPERATURE = 0.4
    TOP_P = 0.9
    MAX_GRAD_NORM = 1.0
    SAVE_PATH = "weights/SmolLM2-135M-torch-grpo"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_response(model, input_ids, max_new_tokens, temperature, top_p, tokenizer):
    """生成响应"""
    model.eval()
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs.logits[:, -1, :] / temperature

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return generated


def simple_scorer(response, expected):
    """简单的评分函数"""
    response_lower = response.lower().strip()
    expected_lower = expected.lower().strip()
    return 1.0 if expected_lower in response_lower else 0.0


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

    print(f"Loading model: {config.MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # 只加载一个模型，使用 PPO 风格的更新
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL,
        torch_dtype=torch.bfloat16 if config.DEVICE == "cuda" else torch.float32,
    ).to(config.DEVICE)

    print(
        f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters"
    )

    train_data = [
        {"prompt": "What is 2+2?", "expected": "4"},
        {"prompt": "What is the capital of France?", "expected": "Paris"},
        {"prompt": "What is 5*5?", "expected": "25"},
    ]

    print(f"Train samples: {len(train_data)}")

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

    model.train()
    all_rewards = []

    for iteration in range(config.ITERS):
        data = random.choice(train_data)
        prompt = data["prompt"]
        expected = data["expected"]

        input_text = prompt
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(config.DEVICE)

        rewards = []
        samples = []

        # 多个采样用于计算奖励
        for i in range(config.GROUP_SIZE):
            generated = model.generate(
                input_ids,
                max_new_tokens=config.GEN_LEN,
                do_sample=True,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated_ids = generated[0][input_ids.shape[0] :]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            reward = simple_scorer(response, expected)
            rewards.append(reward)
            samples.append(generated)
            print(f"  Sample {i}: reward={reward}, response='{response[:50]}...'")

        # 计算平均奖励
        avg_reward = sum(rewards) / len(rewards)
        all_rewards.append(avg_reward)

        # 用最好的样本微调
        best_reward = max(rewards)
        best_idx = rewards.index(best_reward)
        best_sample = samples[best_idx]

        # 计算 loss
        outputs = model(best_sample)
        # 简单负交叉熵作为 loss
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = best_sample[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=tokenizer.pad_token_id,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()

        if (iteration + 1) % 10 == 0:
            print(
                f"Iter {iteration + 1}, Avg Reward: {sum(all_rewards[-10:]) / 10:.4f}, Loss: {loss.item():.4f}"
            )

        if (iteration + 1) % 50 == 0:
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
