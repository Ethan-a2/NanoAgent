# NanoAgent 系统架构分析

## 阶段 2：Mermaid  diagrams

---

## 2.1 组件交互图

```mermaid
flowchart TB
    subgraph Input["输入层"]
        U[用户输入]
        T[工具定义]
        S[系统提示]
    end
    
    subgraph Core["核心模型"]
        E[Tokenizer]
        M[135M LM]
        G[生成器]
    end
    
    subgraph Output["输出层"]
        P[解析器]
        J[JSON Tool Call]
        R[文本回复]
    end
    
    subgraph Tool["工具执行"]
        W[Web Search]
        API[API 调用]
        DB[数据库]
    end
    
    U --> E
    T --> E
    S --> E
    E --> M
    M --> G
    G --> P
    P --> J
    J --> W
    J --> API
    J --> DB
    W --> R
    API --> R
    DB --> R
```

---

## 2.2 数据流图

```mermaid
flowchart LR
    subgraph Preprocess["预处理"]
        C[Chat Template]
        T[Tokenize]
        P[Padding]
    end
    
    subgraph Inference["推理"]
        E[Embed]
        A[Attention]
        F[FFN]
        S[Softmax]
    end
    
    subgraph Postprocess["后处理"]
        D[Decode]
        J[JSON Parse]
        V[Validate]
    end
    
    subgraph ToolExec["工具执行"]
        TOOL[Tool Call]
        EX[Execute]
        RES[Result]
    end
    
    C --> T --> P --> E --> A --> F --> S --> D --> J --> V --> TOOL --> EX --> RES
```

---

## 2.3 序列图：工具调用流程

```mermaid
sequenceDiagram
    participant U as User
    participant M as Model
    participant T as Tokenizer
    participant G as Generator
    participant P as Parser
    participant W as Web Search
    participant E as External API

    U->>M: "What's the latest AI news?"
    M->>T: apply_chat_template(messages)
    T->>T: tokenize(input_text)
    T->>M: input_ids
    M->>M: forward(input_ids)
    M->>G: generate()
    
    rect rgb(240, 248, 255)
        note right of G: Generation Loop
        G->>G: sample next token
        G-->>M: logits
    end
    
    G->>P: generated_text
    P->>P: parse JSON
    P->>W: {"name": "web_search", "arguments": {"query": "AI news"}}
    W->>W: fetch(url)
    W->>E: HTTP Request
    E-->>W: HTML Response
    W-->>P: search_results
    P->>M: results
    M->>M: generate_response(results)
    M-->>U: "The latest AI news is..."
```

---

## 2.4 类图：训练模块

```mermaid
classDiagram
    class TrainConfig {
        <<dataclass>>
        +MODEL: str
        +EPOCHS: int
        +BATCH_SIZE: int
        +CONTEXT_LEN: int
        +MAX_LEARNING_RATE: float
        +SCHEDULER: str
        +SAVE_PATH: str
    }

    class Dataset {
        +dataset: list
        +tokenizer: Tokenizer
        +stride: int
        +__getitem__(idx) -> dict
        +loss_over_instructions() -> weights
    }

    class SFTrainer {
        +model: nn.Module
        +optimizer: optim.Optimizer
        +scheduler: Callable
        +train_step() -> loss
        +eval_step() -> eval_loss
        +save_state()
        +load_state()
    }

    class GRPOTrainer {
        +model: nn.Module
        +model_old: nn.Module
        +group_size: int
        +epsilon: float
        +compute_rewards() -> rewards
        +compute_grpo_loss() -> loss
        +generate_samples() -> samples
    }

    class Sampler {
        +temperature: float
        +top_p: float
        +top_k: int
        +min_p: float
        +sample() -> token
    }

    TrainConfig --> SFTrainer
    TrainConfig --> GRPOTrainer
    Dataset --> SFTrainer
    Sampler --> GRPOTrainer
```

---

## 2.5 序列图：SFT 训练流程

```mermaid
sequenceDiagram
    participant C as Config
    participant D as Dataset
    participant M as Model
    participant O as Optimizer
    participant S as Scheduler
    participant L as Loss
    participant G as Gradient
    participant V as Validator

    C->>C: load config
    D->>D: load_dataset()
    M->>M: load_model()
    O->>O: AdamW()
    S->>S: cosine_decay()

    loop Training Loop
        D->>D: __getitem__(idx)
        D->>V: get_batch()
        V->>M: forward(x)
        M->>L: compute_loss(logits, y)
        L->>G: backprop()
        G->>G: clip_grad_norm()
        G->>O: optimizer.step()
        O->>S: scheduler.step()
        S->>M: update_lr()
    end

    M->>M: save_state()
```

---

## 2.6 序列图：GRPO 训练流程

```mermaid
sequenceDiagram
    participant C as Config
    participant D as Dataset
    participant M as Model
    participant MO as Model Old
    participant G as Generator
    participant R as Reward
    participant L as GRPO Loss
    participant O as Optimizer

    C->>C: load config
    D->>D: load_prompt()
    M->>MO: copy_weights()
    MO->>MO: freeze()

    loop GRPO Iteration
        loop Group Size
            D->>G: generate(prompt)
            G->>R: compute_reward(response)
            R-->>D: reward
        end
        
        R->>L: compute_objectives(rewards)
        L->>L: normalize(rewards)
        L->>L: compute_grpo_loss(objectives)
        L->>L: compute_kl_div()
        
        L->>O: backprop()
        O->>M: update_weights()
        
        alt every N steps
            M->>MO: sync_weights()
        end
    end

    M->>M: save_model()
```

---

## 2.7 状态机：模型生命周期

```mermaid
stateDiagram-v2
    [*] --> BaseModel: download/convert

    state BaseModel {
        [*] --> HFModel: HuggingFace
        HFModel --> MLXModel: convert()
        MLXModel --> Quantized: quantize()
    }

    BaseModel --> SFT: start_sft()

    state SFT {
        [*] --> Init: initialize
        Init --> Warmup: warmup_steps
        Warmup --> Training: training_loop
        Training --> Checkpoint: save_checkpoint
        Checkpoint --> Training: continue
        Training --> Eval: eval_step
        Eval --> Training: continue
        Training --> Done: epoch_complete
    }

    SFT --> SFTModel: save_model()

    SFTModel --> GRPO: start_grpo()

    state GRPO {
        [*] --> InitGroup: init_groups
        InitGroup --> Generate: generate_samples
        Generate --> ComputeReward: reward_fn
        ComputeReward --> Normalize: normalize_rewards
        Normalize --> GRPOLoss: compute_grpo
        GRPOLoss --> Update: optimizer_step
        Update --> Generate: continue
        Update --> Done: iter_complete
    }

    GRPO --> FinalModel: save_model()
    FinalModel --> [*]: export

    SFT --> Eval: evaluate
    GRPO --> Eval: evaluate
```

---

## 2.8 数据流：推理管道

```mermaid
flowchart TB
    subgraph Input["输入"]
        msg[Messages]
        tool[Tools]
    end

    subgraph Template["模板应用"]
        fmt[apply_chat_template]
        gen[add_generation_prompt]
    end

    subgraph Tokenize["分词"]
        enc[encode]
        pad[padding]
    end

    subgraph Generate["生成"]
        mod[forward]
        smp[sample]
        dec[decode]
    end

    subgraph Parse["解析"]
        jsn[json.loads]
        ext[extract_tool]
    end

    subgraph Execute["执行"]
        exec[execute_tool]
        fetch[fetch_result]
    end

    subgraph Feedback["反馈"]
        fb[format_feedback]
        append[append to messages]
    end

    msg --> fmt
    tool --> fmt
    fmt --> gen
    gen --> enc
    enc --> pad
    pad --> mod
    mod --> smp
    smp --> dec
    dec --> jsn
    jsn --> ext
    ext --> exec
    exec --> fetch
    fetch --> fb
    fb --> append
```

---

*Generated by code-insight skill*