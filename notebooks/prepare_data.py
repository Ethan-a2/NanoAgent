from datasets import load_dataset
from transformers import AutoTokenizer

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

# 加载数据
ds = load_dataset("json", data_files="your_data.jsonl", split="train")

# 应用 chat template
def process_example(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=True
    )
    return {"text": text}

ds = ds.map(process_example, batched=False)
ds.to_json("data/datasets/train.jsonl")