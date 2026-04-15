# NanoAgent 系统架构分析

## 阶段 8：详细设计文档

---

## 8.1 系统架构

### 8.1.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NanoAgent                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │    SFT    │───▶│   GRPO    │───▶│  Inference │      │
│  │  Trainer  │    │  Trainer  │    │   Engine  │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│        │                  │                  │                   │
│        ▼                  ▼                  ▼           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │                    MLX Core                        │  │
│  │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐          │  │
│  │  │Embed│  │Attention│ │ FFN │  │Norm │          │  │
│  │  └─────┘  └─────┘  └─────┘  └─────┘          │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.1.2 数据流

```
User Input
    │
    ▼
┌─────────────┐
│  Validator │  # 检查输入格式
└─────────────┘
    │
    ▼
┌─────────────┐
│  Template │  # 应用 Chat Template
└─────────────┘
    │
    ▼
┌─────────────┐
│ Tokenizer │  # 分词
└─────────────┘
    │
    ▼
┌─────────────┐
│   Model   │  # 前向传播
└─────────────┘
    │
    ▼
┌─────────────┐
│ Sampler   │  # 采样
└─────────────┘
    │
    ▼
┌─────────────┐
│  Decode   │  # 解码
└─────────────┘
    │
    ▼
┌─────────────┐
│  Parser   │  # JSON 解析
└─────────────┘
    │
    ▼
┌─────────────┐
│  Execute  │  # 工具执行
└─────────────┘
    │
    ▼
User Response
```

---

## 8.2 模块设计

### 8.2.1 训练模块

#### SFT Trainer

```
┌─────────────────────────────────────┐
│           SFTrainer                 │
├─────────────────────────────────────┤
│                                     │
│  - model: nn.Module                 │
│  - optimizer: optim.Optimizer      │
│  - scheduler: Callable             │
│  - dataset: Dataset             │
│                                     │
│  + train_step() -> loss           │
│  + eval_step() -> eval_loss    │
│  + save_state()                │
│  + load_state()               │
└─────────────────────────────────────┘
```

**职责**：
1. 加载配置和数据集
2. 执行前向/反向传播
3. 更新模型参数
4. 保存检查点

#### GRPO Trainer

```
┌─────────────────────────────────────┐
│          GRPOTrainer               │
├─────────────────────────────────────┤
│                                     │
│  - model: nn.Module                │
│  - model_old: nn.Module           │
│  - group_size: int                │
│  - epsilon: float                │
│                                     │
│  + generate() -> samples         │
│  + compute_reward() -> rewards  │
│  + compute_grpo_loss() -> loss   │
│  + update()                   │
└─────────────────────────────────────┘
```

**职责**：
1. 生成采样轨迹
2. 计算奖励函数
3. 计算 GRPO Loss
4. 更新策略

---

### 8.2.2 推理模块

#### Inference Engine

```
┌─────────────────────────────────────┐
│       InferenceEngine               │
├─────────────────────────────────────┤
│                                     │
│  - model: nn.Module                │
│  - tokenizer: Tokenizer          │
│  - sampler: Sampler           │
│                                     │
│  + generate() -> text      │
│  + stream() -> Generator     │
│  + batch_generate() -> list│
└─────────────────────────────────────┘
```

**职责**：
1. 管理模型和分词器
2. 执行推理
3. 采样生成

#### Tool Executor

```
┌─────────────────────────────────────┐
│         ToolExecutor              │
├─────────────────────────────────────┤
│                                     │
│  - tools: Dict[str, Callable]       │
│  - parser: JSONParser           │
│                                     │
│  + parse() -> List[ToolCall]    │
│  + execute() -> List[Result]    │
│  + format() -> str            │
└─────────────────────────────────────┘
```

**职责**：
1. 解析 JSON 工具调用
2. 执行工具
3. 格式化结果

---

## 8.3 接口设计

### 8.3.1 训练接口

```python
# SFT 训练
def sft_train(
    model_path: str,
    data_path: str,
    save_path: str,
    epochs: int = 1,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    context_len: int = 2048,
):
    """
    SFT 训练接口。

    Args:
        model_path: 基座模型路径
        data_path: 训练数据路径
        save_path: 保存路径
        epochs: 训练轮数
        batch_size: 批量大小
        learning_rate: 学习率
        context_len: 上下文长度
    """
    # 实现...
```

```python
# GRPO 训练
def grpo_train(
    model_path: str,
    data_path: str,
    save_path: str,
    iters: int = 2000,
    group_size: int = 8,
    learning_rate: float = 1e-5,
    gen_len: int = 384,
):
    """
    GRPO 训练接口。

    Args:
        model_path: 模型路径
        data_path: 数据路径
        save_path: 保存路径
        iters: 迭代次数
        group_size: 组大小
        learning_rate: 学习率
        gen_len: 生成长度
    """
    # 实现...
```

### 8.3.2 推理接口

```python
def inference(
    messages: List[Dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.3,
    tools: List[Dict] = None,
) -> str:
    """
    推理接口。

    Args:
        messages: 对话消息
        max_new_tokens: 最大生成 tokens
        temperature: 温度
        tools: 工具定义

    Returns:
        生成文本或工具调用
    """
    # 应用 chat template
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 分词
    inputs = tokenizer.encode(input_text, return_tensors="pt")

    # 生成
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
    )

    # 解码
    response = tokenizer.decode(outputs[0][inputs.shape[1]:])

    return response
```

---

## 8.4 数据设计

### 8.4.1 训练数据格式

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the weather?"},
    {"role": "assistant", "content": "The weather is sunny."}
  ]
}
```

### 8.4.2 GRPO 数据格式

```json
{
  "prompt": "What is 2+2?",
  "response": "4",
  "scorer": "exact_match",
  "reward": 1.0
}
```

### 8.4.3 工具定义格式

```json
{
  "type": "function",
  "function": {
    "name": "web_search",
    "description": "Performs a web search.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The search query."
        }
      },
      "required": ["query"]
    }
  }
}
```

---

## 8.5 配置设计

### 8.5.1 SFT 配置

```python
@dataclass
class SFTConfig:
    """SFT 训练配置"""

    # 模型配置
    MODEL: str = "HuggingFaceTB/SmolLM2-135M"
    CONTEXT_LEN: int = 2048

    # 训练配置
    EPOCHS: int = 1
    BATCH_SIZE: int = 1
    MAX_LEARNING_RATE: float = 1e-4
    MIN_LEARNING_RATE: float = 0
    WARMUP_RATIO: float = 0.05
    SCHEDULER: str = "cosine"
    WEIGHT_DECAY: float = 0.1

    # 优化配置
    GRADIENT_CHECKPOINT_LAYERS: int = None
    QUANTIZATION: int = None

    # 保存配置
    SAVE_PATH: str = "weights/model"
    SAVE_FREQ: int = 500
```

### 8.5.2 GRPO 配置

```python
@dataclass
class GRPOConfig:
    """GRPO 训练配置"""

    # 模型配置
    MODEL: str = "weights/model"
    MAX_INPUT_LEN: int = 384
    GEN_LEN: int = 384

    # 训练配置
    ITERS: int = 2000
    GROUP_SIZE: int = 8
    BATCH_SIZE: int = 1
    LEARNING_RATE: float = 1e-5

    # GRPO 配置
    EPSILON_MIN: float = 0.2
    EPSILON_HIGH: float = 0.272

    # 采样配置
    TEMPERATURE: float = 0.4
    TOP_P: float = 0.9
    TOP_K: int = 50

    # 保存配置
    SAVE_PATH: str = "weights/grpo"
    SAVE_FREQ: int = 50
```

---

## 8.6 安全设计

### 8.6.1 输入验证

```python
def validate_input(messages: List[Dict]) -> bool:
    """验证输入格式"""
    if not messages:
        return False
    if not all("role" in m and "content" in m for m in messages):
        return False
    if not all(m["role"] in ["system", "user", "assistant"] for m in messages):
        return False
    return True
```

### 8.6.2 输出过滤

```python
def filter_output(text: str) -> str:
    """过滤输出"""
    # 移除特殊 token
    text = text.replace("<|im_start|>", "")
    text = text.replace("<|im_end|>", "")
    # 移除空白
    text = text.strip()
    return text
```

### 8.6.3 工具调用限制

```python
def validate_tool_call(tool_call: Dict) -> bool:
    """验证工具调用"""
    if "name" not in tool_call:
        return False
    if "arguments" not in tool_call:
        return False
    # 添加更多验证...
    return True
```

---

## 8.7 性能设计

### 8.7.1 缓存策略

- 使用 KV Cache 加速推理
- 使用 Pickle 缓存数据集
- 使用 Gradient Checkpoint 节省显存

### 8.7.2 并行策略

- 使用 Batch 推理提高吞吐量
- 使用数据并行（需多卡）
- 使用流水线并行（需多卡）

### 8.7.3 量化策略

- 训练时：可选量化
- 推理时：4-bit/8-bit 量化

---

## 8.8 测试设计

### 8.8.1 单元测试

```python
def test_tokenizer():
    """测试分词器"""
    tokenizer = get_tokenizer("HuggingFaceTB/SmolLM2-135M")
    assert tokenizer.eos_token_id == 2
    assert tokenizer.pad_token_id == 0

def test_chat_template():
    """测试 chat template"""
    tokenizer = get_tokenizer("HuggingFaceTB/SmolLM2-135M")
    messages = [{"role": "user", "content": "Hi"}]
    template = tokenizer.apply_chat_template(messages, tokenize=False)
    assert "<|im_start|>user" in template
```

### 8.8.2 集成测试

```python
def test_inference():
    """测试推理"""
    model, tokenizer = load_model()
    messages = [{"role": "user", "content": "Hi"}]
    response = inference(model, tokenizer, messages)
    assert len(response) > 0

def test_tool_call():
    """测试工具调用"""
    tools = [{"type": "function", "function": {...}}]
    messages = [{"role": "system", "content": TOOL_TEMPLATE}]
    messages.append({"role": "user", "content": "Search AI news"})
    response = inference(messages, tools=tools)
    assert json.loads(response)  # 验证 JSON 格式
```

### 8.8.3 压力测试

```python
def test_long_context():
    """测试长上下文"""
    messages = [{"role": "user", "content": "x" * 8000}]
    response = inference(messages, max_new_tokens=256)
    assert len(response) > 0

def test_high_throughput():
    """测试高吞吐量"""
    messages = [{"role": "user", "content": "Hi"}] * 100
    start = time.time()
    for msg in messages:
        inference(msg)
    elapsed = time.time() - start
    print(f"Throughput: {100 / elapsed} req/s")
```

---

*Generated by code-insight skill*
## 附录：使用说明

本文档共分为 8 个阶段：

1. **阶段 1**：概览、架构决策与核心设计 (`phase1-overview.md`)
2. **阶段 2**：Mermaid diagrams (`phase2-mermaid.md`)
3. **���段 3**：核心组件与关键特性 (`phase3-core.md`)
4. **阶段 4**：性能瓶颈与调试 (`phase4-debug.md`)
5. **阶段 5**：入门练习与最小行动 (`phase5-practice.md`)
6. **阶段 6**：开发/维护/扩展最佳实践 (`phase6-best-practice.md`)
7. **阶段 7**：模块问题排查步骤 (`phase7-checklist.md`)
8. **阶段 8**：详细设计文档 (`phase8-detailed.md`)

所有文档保存在 `/media/code/tools/NanoAgent/docs/` 目录下。

---

*Generated by code-insight skill*