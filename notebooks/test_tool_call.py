import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "quwsarohi/NanoAgent-135M"
# MODEL_PATH = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Performs a web search and returns formatted results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        }
    }
]

TOOL_TEMPLATE = """You are a helpful AI assistant. You have tools.

Only execute tools from above. Follow JSON:
```json
[{{"name": "tool_name", "arguments": {{"arg1": "val1"}}}}]
```"""

messages = [
    {"role": "system", "content": TOOL_TEMPLATE.format(tools=json.dumps(tools, indent=2))},
    {"role": "user", "content": "What's the latest AI news?"},
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

outputs = model.generate(
    inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.3,
)

response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print(f"Tool Call Response:\n{response}")

# 怎么输出不是下面的？
# Output: [{"name": "web_search", "arguments": {"query": "latest AI news 2026"}}]
