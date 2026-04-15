# NanoAgent 系统架构分析

## 阶段 3：核心组件与关键特性，核心算法与复杂度

---

## 3.1 核心组件

### 3.1.1 训练模块

#### SFT 训练器 (`sft/train-mlx.py`)

**职责**：
- 加载和转换 HuggingFace 模型到 MLX 格式
- 管理数据集加载和分词
- 执行监督微调训练
- 保存模型检查点

**核心类**：

| 类名 | 职责 |
|------|------|
| `TrainConfig` | 训练超参数配置 |
| `Dataset` | 内存优化的数据集类 |
| `SFTrainer` | 训练循环管理 |

#### GRPO 训练器 (`grpo/grpo-mlx.py`)

**职责**：
- 生成采样轨迹
- 计算奖励函数
- 执行 GRPO 策略更新
- 可视化训练进度

**核心类**：

| 类名 | 职责 |
|------|------|
| `GRPOTrainer` | GRPO 训练主循环 |
| `Sampler` | 采样配置（temperature, top_p 等） |
| `RewardFn` | 奖励计算 |

### 3.1.2 推理模块

#### 分词器 (`utils/tokenizer.py`)

```python
def get_tokenizer(model_path, add_bos=False) -> AutoTokenizer
```

**功能**：
- 加载预训练分词器
- 配置聊天模板（Chat Template）
- 设置特殊 token（bos, eos, pad）

#### 工具模板 (`utils/tokenizer.py:62-116`)

```python
TOOL_TEMPLATE = """You are a helpful AI assistant. You have a set of possible tools..."""
```

### 3.1.3 数据集模块

#### 训练数据集位置

```
data/datasets/
├── Smollm2_base_train_{CONTEXT_LEN}_nemotron_instruct_fc_base.jsonl
├── Smollm2_base_test_{CONTEXT_LEN}_nemotron_instruct_fc_base.jsonl
└── grpo_cache.pickle
```

#### GRPO 数据集生成

```python
# grpo-mlx.py:223-268
train_ds += ifeval_ds(tokenizer, ...)      # 指令跟随
train_ds += salesfores_toolcall(...)     # 工具调用
train_ds += alice_in_wonderland(...)    # 数学推理
train_ds += gsm_symbolic(...)            # GSM
```

---

## 3.2 关键特性

### 3.2.1 梯度检查点（Gradient Checkpointing）

**代码位置**：`utils/utils.py:75-88`

```python
def grad_checkpoint(layer):
    fn = type(layer).__call__

    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)

        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)

    type(layer).__call__ = checkpointed_fn
```

**作用**：减少训练显存占用，用时间换空间

### 3.2.2 动态填充（Dynamic Padding）

**代码位置**：`sft/train-mlx.py:79`

```python
DYNAMIC_PADDING: bool = True
```

**作用**：减少填充 tokens 数量，提高计算效率

### 3.2.3 Cosine 学习率调度

**代码位置**：`sft/train-mlx.py:362-381`

```python
def cosine_decay_with_warmup(
    max_lr: float,
    total_steps: int,
    warmup_steps: int,
    min_lr: float = 0.0,
):
    def schedule(step):
        linear_warmup = max_lr * step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + mx.cos(mx.pi * progress))
        cosine_decay = (max_lr - min_lr) * cosine_decay + min_lr
        return mx.where(step < warmup_steps, linear_warmup, cosine_decay)
    return schedule
```

---

## 3.3 核心算法

### 3.3.1 GRPO（Group Relative Policy Optimization）

**参考**：
- [The Illustrated GRPO](https://abderrahmanskiredjgithub.io/the-illustrated-grpo/)
- [HuggingFace TRL GRPO](https://huggingface.co/docs/trl/main/en/grpo_trainer)

**原理**：

1. **采样**：对同一 prompt 生成 `GROUP_SIZE` 个响应
2. **奖励**：计算每个响应的奖励
3. **归��化**：使用奖励的相对排名计算 objectives
4. **优化**：最大化期望奖励

**代码位置**：`grpo/grpo-mlx.py:550-650`

```python
def compute_grpo_loss(logits, old_logits, rewards, epsilon=0.2):
    """
    GRPO loss:
    - 对同一 prompt 的多个响应分组
    - 基于相对排名分配权重
    - 使用 clip 限制策略变化
    """
    # 1. 计算每个 token 的 log probability
    log_probs = F.log_softmax(logits, dim=-1)
    old_log_probs = F.log_softmax(old_logits, dim=-1)

    # 2. 计算 importance sampling ratio
    ratio = torch.exp(log_probs - old_log_probs)

    # 3. Clip ratio
    ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

    # 4. 加权 rewards
    loss = -ratio * rewards
    return loss.mean()
```

### 3.3.2 DFT（Dynamic Fine-Tuning）

**参考**：[DFT Paper](https://github.com/yongliang-wu/DFT/)

**原理**：逐步增加训练权重，避免灾难性遗忘

```python
def linear_to_one(step: int, total_steps: int, start: float = 0.0) -> float:
    """Linearly increase DFT weight from `start` to 1.0"""
    progress = min(step / total_steps, 1.0)
    return start + (1.0 - start) * progress
```

### 3.3.3 KL Divergence 正则化

**代码位置**：`sft/train-mlx.py:567-575`

```python
def cal_kl_div(_curr_model_logits, x, pad_mask, label_weight=None):
    ref_model_logits = ref_model(x)
    ref_model_logits = mx.stop_gradient(nn.log_softmax(ref_model_logits, axis=-1))
    curr_model_logits = nn.log_softmax(_curr_model_logits, axis=-1)
    kl_div = nn.losses.kl_div_loss(inputs=curr_model_logits, targets=ref_model_logits, axis=-1, reduction="none")
    if label_weight is not None:
        kl_div = kl_div * label_weight
    kl_div = (kl_div * pad_mask).sum() / mx.maximum(pad_mask.sum(), 1e-7)
    return kl_div
```

---

## 3.4 复杂度分析

### 3.4.1 时间复杂度

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| 前向传播 | O(B × L × D) | B=batch, L=seq_len, D=hidden |
| 注意力 | O(B × L² × D) | 标准 transformer 注意力 |
| GRPO 生成 | O(G × B × L × D × T) | G=group_size, T=max_tokens |
| 梯度计算 | O(B × L × D) | 与前向相同 |

### 3.4.2 空间复杂度

| 组件 | 复杂度 | 说明 |
|------|--------|------|
| 模型参数 | 135M × 2 bytes | 约 270MB (FP16) |
| 激活值 | B × L × D × Layers | 可通过 gradient checkpointing 优化 |
| GRPO 缓存 | G × B × L × vocab | 采样缓存 |

### 3.4.3 训练资源

```
Mac M1 (16GB RAM)
├── 批量大小: 1
├── 上下文长度: 2048
├── 梯度检查点: 6 layers
└── 量化: 可选 (4-bit)
```

---

## 3.5 关键配置

### 3.5.1 SFT 配置

```python
@dataclass
class TrainConfig:
    MODEL = "HuggingFaceTB/SmolLM2-135M"
    EPOCHS = 1
    BATCH_SIZE = 1
    CONTEXT_LEN = 1024 * 2
    MAX_LEARNING_RATE = 1e-4
    SCHEDULER = 'cosine'
    WEIGHT_DECAY = 0.1
    DFT_WEIGHT = 0
    KL_DIV_WEIGHT = 0
```

### 3.5.2 GRPO 配置

```python
@dataclass
class GRPOConfig:
    ITERS = 2_000
    GROUP_SIZE = 8
    GEN_LEN = 384
    LEARNING_RATE = 1e-5
    EPSILON_MIN = 0.2
    EPSILON_HIGH = 0.272
    TEMPERATURE = 0.4
    TOP_P = 0.9
```

---

## 3.6 评估基准

### IFEval（指令跟随）

```python
# ifeval_ds() 生成指令跟随数据集
# 评估约束满足率
```

### BFCL（工具调用）

```python
# BFCL 基准测试
# 评估 JSON 解析准确率
# 整体准确率: 28.99%
```

---

*Generated by code-insight skill*