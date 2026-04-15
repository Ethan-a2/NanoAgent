# NanoAgent 系统架构分析

## 阶段 5：入门练习与最小行动

---

## 5.1 快速开始

### 5.1.1 环境安装

```bash
# 1. 克隆仓库
git clone https://github.com/QuwsarOhi/NanoAgent.git
cd NanoAgent

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装 MLX (macOS)
pip install mlx
pip install mlx-lm
```

### 5.1.2 推理测试

```python
# notebooks/inference.ipynb
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "quwsarohi/NanoAgent-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

def inference(messages, max_new_tokens=256, temperature=0.3):
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
    )
    return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

messages = [{"role": "user", "content": "Hi! Do you have a name?"}]
print(inference(messages))
```

### 5.1.3 工具调用测试

```python
import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Performs a web search.",
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
response = inference(messages, max_new_tokens=512)
print(response)
```

---

## 5.2 数据集准备

### 5.2.1 训练数据格式

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

### 5.2.2 数据处理脚本

```python
# data/dataprep.py
from datasets import load_dataset

# 加载数据集
ds = load_dataset("json", data_files="data.jsonl", split="train")

# 预处理
def preprocess(examples):
    # 应用 chat template
    texts = tokenizer.apply_chat_template(
        examples["messages"], tokenize=False
    )
    return {"text": texts}

ds = ds.map(preprocess, batched=False)
ds.to_json("train.jsonl")
```

---

## 5.3 训练实验

### 5.3.1 SFT 训练

```bash
# Mac M1 训练
python sft/train-mlx.py
```

**配置修改**：
```python
# sft/train-mlx.py:29-54
@dataclass
class TrainConfig:
    MODEL = "HuggingFaceTB/SmolLM2-135M"
    EPOCHS = 1
    BATCH_SIZE = 1
    CONTEXT_LEN = 1024 * 2
    MAX_LEARNING_RATE = 1e-4
    SCHEDULER = 'cosine'
    SAVE_PATH = "weights/my-model"
```

### 5.3.2 GRPO 训练

```bash
# GRPO 训练
python grpo/grpo-mlx.py
```

**配置修改**：
```python
# grpo/grpo-mlx.py:57-93
@dataclass
class TrainConfig:
    ITERS = 2000
    GROUP_SIZE = 8
    GEN_LEN = 384
    LEARNING_RATE = 1e-5
    TEMPERATURE = 0.4
    SAVE_PATH = "weights/my-grpo-model"
```

---

## 5.4 评估实验

### 5.4.1 IFEval 评估

```bash
# 运行 IFEval
lm_eval model \
    --model_path weights/my-model \
    --tasks ifeval \
    --output_path results/
```

### 5.4.2 BFCL 评估

```bash
# 运行 BFCL
python benchmarks/bfcl/bfcl_eval.py \
    --model_path weights/my-model \
    --output results/bfcl/
```

---

## 5.5 最小行动清单

### 5.5.1 第一步：环境搭建

- [ ] 安装 Python 依赖
- [ ] 安装 MLX (macOS)
- [ ] 下载基座模型

### 5.5.2 第二步：推理验证

- [ ] 测试基本对话
- [ ] 测试工具调用
- [ ] 验证 JSON 输出

### 5.5.3 第三步：数据准备

- [ ] 准备训练数据（JSONL 格式）
- [ ] 应用 chat template
- [ ] 数据去重

### 5.5.4 第四步：训练实验

- [ ] 运行 SFT 训练
- [ ] 评估模型性能
- [ ] 调整超参数

### 5.5.5 第五步：GRPO 强化（可选）

- [ ] 准备 GRPO 数据集
- [ ] 运行 GRPO 训练
- [ ] 评估工具调用

---

## 5.6 常见错误

### 5.6.1 依赖安装错误

```bash
# 错误
pip install mlx  # 仅支持 macOS

# 正确 (使用 conda)
conda install -c conda-forge mlx
# 或从源码编译
git clone https://github.com/ml-explore/mlx.git
cd mlx && pip install -e .
```

### 5.6.2 模型加载错误

```python
# 错误
model = AutoModelForCausalLM.from_pretrained("model_path")

# 正确 - 指定 dtype
model = AutoModelForCausalLM.from_pretrained(
    "model_path",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

### 5.6.3 Tokenizer 错误

```python
# 错误 - 未设置特殊 token
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 正确 - 使用 get_tokenizer
from utils.tokenizer import get_tokenizer
tokenizer = get_tokenizer(model_path)
print(tokenizer.eos_token_id)  # 验证
```

---

## 5.7 扩展任务

### 5.7.1 添加新工具

```python
# 在 utils/tools.py 添加新工具
def new_tool(query):
    """New tool description."""
    # 实现逻辑
    return result

# 更新 TOOL_TEMPLATE
tools = [
    {
        "type": "function",
        "function": {
            "name": "new_tool",
            "description": "New tool description.",
            "parameters": {...}
        }
    }
]
```

### 5.7.2 微调其他模型

```python
# 修改 sft/train-mlx.py:32
@dataclass
class TrainConfig:
    MODEL = "HuggingFaceTB/SmolLM2-135M"  # 改为其他模型
```

---

*Generated by code-insight skill*