# NanoAgent 系统架构分析

## 阶段 6：开发/维护/扩展最佳实践

---

## 6.1 代码组织

### 6.1.1 目录结构

```
NanoAgent/
├── sft/                    # SFT 训练
│   └── train-mlx.py
├── grpo/                   # GRPO 训练
│   ├── grpo-mlx.py
│   ├── grpo-trl.py
│   └── grpo-autostart.py
├── utils/                  # 工具函数
│   ├── tokenizer.py
│   ├── tools.py
│   ├── utils.py
│   └── gguf_conv.py
├── benchmarks/             # 评估
│   └── bfcl/
├── config/                # 配置
├── data/                  # 数据处理
└── docs/                 # 文档
```

### 6.1.2 命名规范

- **变量**：`snake_case`（如 `max_learning_rate`）
- **类名**：`PascalCase`（如 `TrainConfig`）
- **常量**：`UPPER_CASE`（如 `MAX_INPUT_LEN`）
- **文件**：`snake_case.py`（如 `train_mlx.py`）

---

## 6.2 开发规范

### 6.2.1 配置管理

```python
# 使用 dataclass 集中配置
from dataclasses import dataclass

@dataclass
class TrainConfig:
    MODEL: str = "HuggingFaceTB/SmolLM2-135M"
    EPOCHS: int = 1
    BATCH_SIZE: int = 1
```

### 6.2.2 错误处理

```python
# 使用断言检查配置
assert TrainConfig.DFT_WEIGHT > 0, "DFT weight must be positive"
assert 0 <= TrainConfig.WEIGHT_DECAY, "Weight decay must be non-negative"
```

### 6.2.3 日志输出

```python
# 结构化日志
import json
config_dict = {
    k: v for k, v in TrainConfig.__dict__.items()
    if not k.startswith("__") and not callable(v)
}
print(json.dumps(config_dict, indent=2))
```

---

## 6.3 训练最佳实践

### 6.3.1 学习率设置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| Max LR | 1e-4 | SFT 推荐值 |
| Min LR | 0 | 防止遗忘 |
| Warmup Ratio | 0.05 | 5% 预热 |
| Scheduler | cosine | 平滑收敛 |

### 6.3.2 训练技巧

1. **梯度检查点**
   ```python
   # sft/train-mlx.py:110-114
   if TrainConfig.GRADIENT_CHECKPOINT_LAYERS is not None:
       for layer in model.layers[:nlayers]:
           grad_checkpoint(layer)
   ```

2. **梯度裁剪**
   ```python
   # sft/train-mlx.py:628
   clipped_grads, total_norm = optim.clip_grad_norm(grads, max_norm=1.0)
   ```

3. **Early Stopping**
   ```python
   # 监控验证集 loss
   if eval_loss < best_eval_loss:
       best_eval_loss = eval_loss
       save_checkpoint()
   ```

---

## 6.4 推理最佳实践

### 6.4.1 采样配置

```python
def inference(messages, max_new_tokens=256, **kwargs):
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.3,      # 推荐: 0.2-0.4
        top_p=0.9,        # 推荐: 0.9
        repetition_penalty=1.05,  # 推荐: 1.0-1.1
    )
```

### 6.4.2 工具调用格式

```python
# 使用 JSON 格式
TOOL_TEMPLATE = """You are a helpful AI assistant. You have tools.

Only execute tools from above. Follow JSON:
```json
[{{"name": "tool_name", "arguments": {{"arg1": "val1"}}}}]
```"""
```

---

## 6.5 数据最佳实践

### 6.5.1 数据质量

1. **去重**
   ```python
   # 移除重复数据
   dataset = dataset.remove_duplicates()
   ```

2. **过滤**
   ```python
   # 过滤短响应
   dataset = dataset.filter(lambda x: len(x["response"]) > 10)
   ```

3. **格���化**
   ```python
   # 标准化格式
   def normalize(example):
       example["messages"] = format_messages(example["messages"])
       return example
   ```

### 6.5.2 数据存储

- 使用 **JSONL** 格式存储训练数据
- 使用 **Pickle** 缓存处理后的数据
- 定期备份数据集

---

## 6.6 模型维护

### 6.6.1 版本管理

```bash
# 使用 git tag
git tag -a v1.0 -m "Release v1.0"
git push origin v1.0
```

### 6.6.2 检查点管理

```python
# sft/train-mlx.py:488-521
def save_state(iter_step, losses, model, optimizer, path):
    # 保存模型
    save_model(save_path=path, model=dequantize_model(model))
    # 保存优化器状态
    mx.save_safetensors(
        os.path.join(path, "optimizer.safetensors"),
        dict(tree_flatten(optimizer.state))
    )
    # 保存训练信息
    with open(os.path.join(path, "train_info.json"), "w") as f:
        json.dump(train_info, f)
```

### 6.6.3 模型导出

```python
# 导出为 HuggingFace 格式
tokenizer.save_pretrained(path)
model.save_pretrained(path)

# 导出为 GGUF
python utils/gguf_conv.py --input weights/model --output weights/model.gguf
```

---

## 6.7 测试最佳实践

### 6.7.1 单元测试

```python
# tests/test_tokenizer.py
def test_tokenizer():
    tokenizer = get_tokenizer("HuggingFaceTB/SmolLM2-135M")
    assert tokenizer.eos_token_id == 2
    assert tokenizer.pad_token_id == 0
    
def test_chat_template():
    messages = [{"role": "user", "content": "Hi"}]
    template = tokenizer.apply_chat_template(messages, tokenize=False)
    assert "<|im_start|>user" in template
```

### 6.7.2 集成测试

```python
# tests/test_inference.py
def test_inference():
    model, tokenizer = load_model()
    messages = [{"role": "user", "content": "Hi"}]
    response = inference(model, tokenizer, messages)
    assert len(response) > 0
```

---

## 6.8 扩展指南

### 6.8.1 添加新数据集

```python
# data/grpo/my_dataset.py
def my_dataset(tokenizer, size=1000):
    dataset = []
    for _ in range(size):
        prompt, response = generate_sample()
        dataset.append({
            "prompt": prompt,
            "response": response,
            "scorer": scorer_function,
        })
    return dataset
```

### 6.8.2 添加新工具

```python
# utils/tools.py
def new_tool(param1: str, param2: int) -> dict:
    """Tool description."""
    result = call_api(param1, param2)
    return {"result": result}
```

### 6.8.3 添加新训练方法

```python
# 新建训练脚本
# 参考 sft/train-mlx.py 和 grpo/grpo-mlx.py
# 实现自定义训练循环
```

---

*Generated by code-insight skill*