# NanoAgent 使用指南 (PyTorch CUDA 版)

本文档详细介绍 NanoAgent 项目中所有脚本的使用方法。

---

## 目录

1. [环境准备](#1-环境准备)
2. [模型推理](#2-模型推理)
3. [数据准备](#3-数据准备)
4. [模型训练](#4-模型训练)
5. [模型评估](#5-模型评估)
6. [工具脚本](#6-工具脚本)

---

## 1. 环境准备

### 1.1 基础依赖

```bash
# 克隆仓库
git clone https://github.com/QuwsarOhi/NanoAgent.git
cd NanoAgent

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装 Python 依赖
pip install -r requirements.txt
```

### 1.2 CUDA 环境配置

```bash
# 检查 CUDA 版本
nvcc --version

# 检查 GPU
nvidia-smi
```

安装 PyTorch CUDA 版本：

```bash
# 安装 PyTorch (CUDA 12.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 transformers (支持 CUDA)
pip install transformers accelerate bitsandbytes
```

### 1.3 验证安装

```bash
# 检查 Python 版本
python --version
# 期望: Python 3.9+

# 检查 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 检查 GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## 2. 模型推理

### 2.1 使用 Jupyter Notebook

```bash
# 启动 Jupyter
jupyter notebook notebooks/inference.ipynb
```

使用 `notebooks/inference.ipynb` 中的代码进行推理。

### 2.2 使用 Python 脚本

创建 `test_inference.py`：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型路径（本地或 HuggingFace）
MODEL_PATH = "quwsarohi/NanoAgent-135M"

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
```

运行：

```bash
python test_inference.py
```

### 2.3 工具调用推理

创建 `test_tool_call.py`：

```python
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "quwsarohi/NanoAgent-135M"
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
```

运行：

```bash
python test_tool_call.py
```

---

## 3. 数据准备

### 3.1 训练数据格式

训练数据采用 JSONL 格式：

```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### 3.2 处理数据

创建 `prepare_data.py`：

```python
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
```

运行：

```bash
python prepare_data.py
```

---

## 4. 模型训练

### 4.1 SFT 监督微调

#### 4.1.1 配置修改

编辑 `sft/train-torch.py` 中的 `TrainConfig`：

```python
@dataclass
class TrainConfig:
    MODEL = "HuggingFaceTB/SmolLM2-135M"  # 基座模型
    EPOCHS = 1                            # 训练轮数
    BATCH_SIZE = 4                       # 批量大小 (根据显存调整)
    CONTEXT_LEN = 2048                   # 上下文长度
    LEARNING_RATE = 1e-4                 # 学习率
    WEIGHT_DECAY = 0.1                   # 权重衰减
    SAVE_PATH = "weights/my-model"       # 保存路径
```

#### 4.1.2 运行训练

```bash
# 使用 GPU 训练
python sft/train-torch.py
```

训练过程中会显示：

```
Step 100, Loss: 2.4532
Step 200, Loss: 2.1234
...
```

### 4.2 GRPO 强化训练

#### 4.2.1 配置修改

编辑 `grpo/grpo-torch.py` 中的 `TrainConfig`：

```python
@dataclass
class TrainConfig:
    MODEL = "weights/my-sft-model"    # SFT 模型路径
    ITERS = 2000                    # 迭代次数
    GROUP_SIZE = 8                  # 组大小
    LEARNING_RATE = 1e-5           # 学习率
    TEMPERATURE = 0.4               # 采样温度
    TOP_P = 0.9                    # Top-p 采样
    SAVE_PATH = "weights/my-grpo-model"
```

#### 4.2.2 运行训练

```bash
python grpo/grpo-torch.py
```

---

## 5. 模型评估

### 5.1 BFCL 工具调用评估

```bash
# 安装 BFCL
pip install bfcl_eval

# 运行评估
python benchmarks/bfcl/bfcl_eval.py \
    --model_path weights/my-model \
    --output results/bfcl/
```

### 5.2 IFEval 评估

```bash
# 安装 lm-evaluation-harness
pip install lm-evaluation-harness

# 运行评估
lm_eval model \
    --model_path weights/my-model \
    --tasks ifeval \
    --output_path results/ifeval/
```

### 5.3 自定义评估

创建 `custom_eval.py`：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "weights/my-model"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

test_cases = [
    {"input": "Hi!", "expected": "Hello"},
    {"input": "What is 2+2?", "expected": "4"},
]

correct = 0
for case in test_cases:
    messages = [{"role": "user", "content": case["input"]}]
    # ... 运行推理
    if response == case["expected"]:
        correct += 1

accuracy = correct / len(test_cases)
print(f"Accuracy: {accuracy:.2%}")
```

---

## 6. 工具脚本

### 6.1 基础推理脚本

```bash
# 直接运行
python notebooks/inference.py
```

### 6.2 内存监控

```bash
python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB')
print(f'Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB')
"
```

### 6.3 模型信息

```bash
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('weights/my-model', torch_dtype=torch.float16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('weights/my-model')

param_count = sum(p.numel() for p in model.parameters())
print(f'Parameters: {param_count / 1e6:.1f}M')
"
```

---

## 常见问题

### Q: 显存不足怎么办？

A:
1. 减小 `BATCH_SIZE`
2. 减小 `CONTEXT_LEN`
3. 使用梯度累积：`GRADIENT_ACCUMULATION_STEPS`
4. 使用 8-bit 量化：`load_in_8bit=True`

### Q: 训练速度慢怎么办？

A:
1. 增大 `BATCH_SIZE`
2. 使用混合精度：`torch.cuda.amp`
3. 使用梯度累积
4. 检查 CUDA 设置

### Q: 模型不收敛怎么办？

A:
1. 检查学习率（建议 1e-4 ~ 1e-5）
2. 检查数据是否正确加载
3. 查看是否有 NaN 梯度

### Q: 工具调用失败怎么办？

A:
1. 降低 temperature（0.1 ~ 0.3）
2. 检查 prompt 模板格式
3. 添加示例

---

## 附录：脚本对照表

| 脚本 | 用途 | 设备 |
|------|------|------|
| `sft/train-torch.py` | SFT 训练 | CUDA |
| `grpo/grpo-torch.py` | GRPO 训练 | CUDA |
| `sft/train-mlx.py` | SFT 训练 | Apple MLX |
| `grpo/grpo-mlx.py` | GRPO 训练 | Apple MLX |

---

## 参考

- [NanoAgent HuggingFace](https://huggingface.co/quwsarohi/NanoAgent-135M)
- [PyTorch 文档](https://pytorch.org/docs/)
- [Transformers 文档](https://huggingface.co/docs/transformers)