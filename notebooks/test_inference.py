import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型路径（本地或 HuggingFace）
MODEL_PATH = "quwsarohi/NanoAgent-135M"
# MODEL_PATH = "HuggingFaceTB/SmolLM2-135M-Instruct"


# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,  # 使用半精度
    device_map="auto"          # 自动分配到 GPU
)

def inference(messages, max_new_tokens=256, temperature=0.3):
    """推理函数"""
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.05,
    )
    
    return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

# 测试基本对话
messages = [{"role": "user", "content": "Hi! Do you have a name?"}]
response = inference(messages)
print(f"User: {messages[0]['content']}")
print(f"Assistant: {response}")