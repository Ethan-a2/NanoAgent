# NanoAgent 使用指南

本文档详细介绍 NanoAgent 项目中所有脚本的使用方法。

---

## 目录

1. [环境准备](#1-环境准备)
2. [模型推理](#2-模型推理)
3. [数据准备](#3-数据准备)
4. [模型训练](#4-模型训练)
5. [模型评估](#5-模型评估)
6. [模型转换与导出](#6-模型转换与导出)
7. [工具脚本](#7-工具脚本)

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

### 1.2 MLX 安装（仅 macOS）

```bash
# 使用 pip 安装
pip install mlx

# 或使用 conda
conda install -c conda-forge mlx

# 安装 mlx-lm
pip install mlx-lm
```

### 1.3 验证安装

```bash
# 检查 Python 版本
python --version
# 期望: Python 3.9+

# 检查 MLX
python -c "import mlx; print(mlx.__version__)"

# 检查 transformers
python -c "import transformers; print(transformers.__version__)"
```

---

## 2. 模型推理

### 2.1 使用 Jupyter Notebook（推荐）

```bash
# 启动 Jupyter
jupyter notebook notebooks/inference.ipynb
```

在 notebook 中运行 cells 即可进行推理。

### 2.2 使用 Python 脚本

创建 `test_inference.py`：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# 模型路径（本地或 HuggingFace）
MODEL_PATH = "quwsarohi/NanoAgent-135M"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
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
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "quwsarohi/NanoAgent-135M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")

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

# 系统提示模板
TOOL_TEMPLATE = """You are a helpful AI assistant. You have a set of possible tools that you can execute to retrieve information or to perform specific actions. You can execute zero or more tools to answer user question.

Here are the list of tools that you have access to:
```json
{tools}
```

Only execute tools from above. Follow the below JSON signature to execute tools:
```json
[{{"name": "tool_name", "arguments": {{"arg1": "val1", ...}}}}, ...]
```
"""

# 构建消息
messages = [
    {"role": "system", "content": TOOL_TEMPLATE.format(tools=json.dumps(tools, indent=2))},
    {"role": "user", "content": "What's the latest AI news?"},
]

# 推理
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

### 3.1 准备训练数据

数据准备脚本位于 `data/dataprep.py`，需要配置数据集来源后运行：

```bash
# 查看可用的数据集准备函数
grep "^def " data/dataprep.py
```

常用函数：

- `shortcodes_python()` - Python 代码数据
- `orca_math()` - 数学推理数据
- `tool_calling()` - 工具调用数据

### 3.2 数据格式

训练数据采用 JSONL 格式：

```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### 3.3 处理已有数据

创建 `prepare_data.py`：

```python
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from utils.tokenizer import get_tokenizer

# 加载数据
ds = load_dataset("json", data_files="your_data.jsonl", split="train")

# 加载分词器
tokenizer = get_tokenizer("HuggingFaceTB/SmolLM2-135M")

# 应用 chat template
def process_example(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=True
    )
    return {"text": text}

ds = ds.map(process_example, batched=False)

# 保存
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

编辑 `sft/train-mlx.py` 中的 `TrainConfig`：

```python
@dataclass
class TrainConfig:
    MODEL = "HuggingFaceTB/SmolLM2-135M"        # 基座模型
    EPOCHS = 1                                   # 训练轮数
    BATCH_SIZE = 1                               # 批量大小
    CONTEXT_LEN = 2048                          # 上下文长度
    MAX_LEARNING_RATE = 1e-4                      # 最大学习率
    SCHEDULER = 'cosine'                        # 学习率调度器
    SAVE_PATH = "weights/my-model"               # 保存路径
```

#### 4.1.2 运行训练

```bash
# Mac M1/M2/M3
python sft/train-mlx.py
```

训练过程中会显示：
```
TL 2.4532|0.0001 / EL 2.1234|0.0001 / CTX (2048, 1847, 2048) | LR 0.000095
Training progress: 25.00
```

#### 4.1.3 从检查点恢复

```python
@dataclass
class TrainConfig:
    LOAD_PREV = True  # 设置为 True
```

### 4.2 GRPO 强化训练

#### 4.2.1 配置修改

编辑 `grpo/grpo-mlx.py` 中的 `TrainConfig`：

```python
@dataclass
class TrainConfig:
    MODEL = "weights/my-sft-model"              # SFT 模型路径
    ITERS = 2000                             # 迭代次数
    GROUP_SIZE = 8                           # 组大小
    LEARNING_RATE = 1e-5                      # 学习率
    TEMPERATURE = 0.4                        # 采样温度
    TOP_P = 0.9                              # Top-p 采样
    SAVE_PATH = "weights/my-grpo-model"         # 保存路径
    GENERATE_DATA = True                      # 是否生成数据
```

#### 4.2.2 运行训练

```bash
python grpo/grpo-mlx.py
```

训练过程中会：
1. 生成训练数据集
2. 对每个 prompt 采样 `GROUP_SIZE` 个响应
3. 计算奖励并更新策略
4. 绘制训练曲线

### 4.3 使用 TRL 库训练

```bash
# 使用 HuggingFace TRL
python grpo/grpo-trl.py
```

---

## 5. 模型评估

### 5.1 IFEval 评估

```bash
# 安装 lm-evaluation-harness
pip install lm-evaluation-harness

# 运行评估
lm_eval model \
    --model_path weights/my-model \
    --tasks ifeval \
    --output_path results/ifeval/
```

### 5.2 BFCL 工具调用评估

#### 5.2.1 基本评估

```bash
python benchmarks/bfcl/bfcl_eval.py \
    --model_path weights/my-model \
    --output results/bfcl/
```

#### 5.2.2 评估特定类别

```bash
# 仅评估 simple_python
python benchmarks/bfcl/bfcl_eval.py \
    --model_path weights/my-model \
    --category simple_python \
    --output results/bfcl/
```

#### 5.2.3 运行测试脚本

```bash
# 测试推理
python benchmarks/bfcl/test_inference.py

# 测试多个模型
python benchmarks/bfcl/test_multiple.py
```

### 5.3 自定义评估

创建 `custom_eval.py`：

```python
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "weights/my-model"

# 加载模型
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 测试用例
test_cases = [
    {"input": "Hi!", "expected": "Hello"},
    {"input": "What is 2+2?", "expected": "4"},
]

# 运行评估
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

## 6. 模型转换与导出

### 6.1 转换为 MLX 格式

```python
from mlx_lm.utils import convert

# HuggingFace -> MLX
convert("HuggingFaceTB/SmolLM2-135M", mlx_path="weights/SmolLM2-135M")
```

### 6.2 导出为 GGUF

```bash
python utils/gguf_conv.py \
    --input weights/my-model \
    --output weights/my-model.gguf \
    --quantization q4_0
```

支持的量化方式：
- `q4_0` - 4-bit 量化
- `q5_0` - 5-bit 量化
- `q8_0` - 8-bit 量化
- `f16` - 半精度
- `f32` - 全精度

### 6.3 导出为 HuggingFace 格式

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载 MLX 模型并导出
model = AutoModelForCausalLM.from_pretrained("weights/my-model")
tokenizer = AutoTokenizer.from_pretrained("weights/my-model")

# 保存
model.save_pretrained("weights/exported")
tokenizer.save_pretrained("weights/exported")
```

### 6.4 上传到 HuggingFace

```bash
# 登录
huggingface-cli login

# 上传模型
huggingface-cli upload your-username/NanoAgent-135M \
    weights/my-model \
    --repo-type model
```

---

## 7. 工具脚本

### 7.1 分词器工具

```bash
# 测试分词器
python -c "
from utils.tokenizer import get_tokenizer
t = get_tokenizer('HuggingFaceTB/SmolLM2-135M')
print('EOS:', t.eos_token_id)
print('PAD:', t.pad_token_id)
"
```

### 7.2 模型信息

```bash
python -c "
from mlx_lm import load
model, tokenizer = load('weights/my-model')
print(model)
"
```

### 7.3 内存监控

```bash
python -c "
import mlx.core as mx
print(f'Active: {mx.get_active_memory() / 1024 / 1024:.2f} MB')
print(f'Peak: {mx.get_peak_memory() / 1024 / 1024:.2f} MB')
"
```

---

## 常见问题

### Q: 显存不足怎么办？

A: 
1. 减小 `CONTEXT_LEN`
2. 启用梯度检查点：`GRADIENT_CHECKPOINT_LAYERS = 6`
3. 使用量化：`QUANTIZATION = 4`

### Q: 训练很慢怎么办？

A:
1. 增大 `BATCH_SIZE`（需要更多显存）
2. 检查是否为量化模型训练
3. 使用 SSD 存储数据

### Q: 模型不收敛怎么办？

A:
1. 检查学习率（建议 1e-4 ~ 1e-5）
2. 检查数据是否正确加载
3. 查看是否有 NaN 梯度

### Q: 工具调用失败怎么办？

A:
1. 降低 temperature（0.1 ~ 0.3）
2. 检查 prompt 模板格式
3. 添加示例到 system prompt

---

## 附录：目录结构

```
NanoAgent/
├── sft/                          # SFT 训练
│   └── train-mlx.py
├── grpo/                         # GRPO 训练
│   ├── grpo-mlx.py
│   ├── grpo-trl.py
│   └── grpo-autostart.py
├── utils/                        # 工具函数
│   ├── tokenizer.py
│   ├── tools.py
│   ├── utils.py
│   └── gguf_conv.py
├── data/                        # 数据处理
│   ├── dataprep.py
│   ├── utils.py
│   └── grpo/
├── benchmarks/                  # 评估
│   └── bfcl/
├── notebooks/                  # Jupyter notebooks
│   ├── inference.ipynb
│   └── test.ipynb
├── config/                    # 配置
└── weights/                   # 模型权重
```

---

## 参考

- [NanoAgent HuggingFace](https://huggingface.co/quwsarohi/NanoAgent-135M)
- [MLX 文档](https://ml-explore.github.io/mlx/)
- [mlx-lm 文档](https://github.com/ml-explore/mlx-lm)