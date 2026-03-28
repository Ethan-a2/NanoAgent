# 🧠 NanoAgent — A 135M Parameter Agentic SLM

NanoAgent is a **135M parameter**, **8k context length**, open-source language model designed for **agentic tasks** such as **tool calling**, **instruction following**, and **lightweight reasoning**.  
It’s small enough (~135 MB in 8-bit) to run on **edge devices** like personal laptops, low-memory CPUs, and even wearables — yet smart enough to make tool calls, parse web information, and give structured answers.

Quick inference resource: [here](notebooks/inference.ipynb)

Huggingface Model: [NanoAgent-135M](https://huggingface.co/quwsarohi/NanoAgent-135M)

Run in Ollama: `ollama run quwsarohi/NanoAgent`

## 🌍 Real-World Use Cases

- 🕹️ **Runs on edge devices** — laptops, smartwatches, browsers, or CPU-only environments.  
- 🌐 **Parses and answers from the web** — supports tool calling to fetch real-time information.  
- 🔎 **Answers recent questions** with live web search tools.  
- 💬 **Continues conversations** — ideal for assistant or agent frameworks.  
- ⚙️ **Tool calling support** enables chaining multiple tools and parsing results to produce final answers.


## ✨ What NanoAgent Supports

| Capability                        | Description                                                                                     | 
|------------------------------------|--------------------------------------------------------------------------------------------------|
| 💬 Basic conversation              | Casual small talk                                                                     |
| 🌐 Information retrieval           | e.g., *“How to bake a cake?”*, *“Weather in Toronto”* through web search. Extracts answers from information returned by tools (scraping/search)                        |
| 🧰 Tool calling                    | Single & multi-tool call with structured explanation                                            |
| 🧠 Question decomposition          | Breaks complex questions into steps                                                             | 
| 🧭 Question classification         | Identifies type of user query (e.g., fact, reasoning, instruction)                              |
| 📝 Following system prompts       | Responds properly to system-level instructions                                                  | 
| ✍️ Writing emails and tasks       | Writes emails, structured messages                                                              | 
---

## 🧪 Training Overview

- **Base model**: [`SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) (instruction-tuned)
- **Fine-tuning method**: ~~[Dynamic Fine-Tuning (DFT)](https://github.com/yongliang-wu/DFT/tree/master)~~ Supervised Fine-Tuning
- **Platform**: Apple Mac M1 (16 GB) — MLX framework

### 📚 Datasets Used

This model was trained using a combination of datasets under different open licenses.  
Each dataset retains its original license, and use of those datasets is subject to their respective terms.

#### General Training (SFT)
| Dataset | Purpose | License |
|---------|---------|---------|
| [microsoft/orca-math-word-problems-200k](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | Math reasoning, word-level reasoning | MIT |
| [allenai/tulu-3-sft-personas-instruction-following](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following) | Instruction following with personas | Open Data Commons License Attribution |
| [mlabonne/orca-agentinstruct-1M-v1-cleaned](https://huggingface.co/datasets/mlabonne/orca-agentinstruct-1M-v1-cleaned) | RAG, MCQ, JSON parsing, text classification | Community Data License Agreement – Permissive, Version 2.0 |
| [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) (systemchats-30k) | General conversation, system prompts | Apache-2.0 |
| [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) (everyday-conversations) | Everyday conversation | Apache-2.0 |
| [nvidia/Nemotron-Instruction-Following-Chat-v1](https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1) | Instruction following, structured outputs | NVIDIA Open Model License |

#### Function Calling Training
| Dataset | Purpose | License |
|---------|---------|---------|
| [Locutusque/function-calling-chatml](https://huggingface.co/datasets/Locutusque/function-calling-chatml) | Tool call response formatting | Apache-2.0 |
| [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | Function calling coverage | Creative Commons Attribution 4.0 |
| [nemotron/interactive_agent](https://huggingface.co/datasets/nemotron/interactive_agent) (local) | Tool calling, agentic behavior | NVIDIA Open Model License |


## 🧭 Key Explorations & Findings

- ✂️ **Dataset deduplication** significantly improved performance by removing noisy or duplicate Q/As.  
 - ✂️ **Shortening the responses** (casual response) and using shorter python code in training improved performance and reduce repeated token generation.
- 🧮 **Word-level reasoning** from `orca-math` enhanced the model’s ability to handle stepwise logic.  
- 🧰 Designing tool calling prompts using **six open-source tool calling datasets** resulted in stronger structured output generation.  
- 🌐 Tool calling integration enabled the model to **extract answers from parsed web data**, supporting up-to-date queries.  


## ⚡ Benchmark

### Model Comparison

| Benchmark | SmolLM2-135M-Instruct | NanoAgent-v0.1 | NanoAgent-v0.2 |
|-----------|:---------------------:|:--------------:|:--------------:|
| **Commonsense QA** (acc) | 20.88% | 20.72% | 20.23% |
| **IFEval** (prompt strict) | 21.63% | 24.58% | **29.94%** |
| **IFEval** (inst strict) | 35.01% | 37.89% | **42.33%** |
| **IFEval** (prompt loose) | 23.84% | 26.80% | **32.16%** |
| **IFEval** (inst loose) | 37.65% | 40.05% | **45.32%** |
| **tinyArc** (acc_norm) | 33.76% | **38.25%** | 36.47% |
| **tinyGSM8k** (exact_match) | 0.55% | **5.95%** | 2.31% |
| **tinyHellaswag** (acc_norm) | 42.20% | 40.41% | **43.45%** |
| **tinyMMLU** (acc_norm) | 26.79% | 25.30% | **27.62%** |
| **tinyTruthfulQA** (acc) | 38.65% | 38.90% | **40.45%** |
| **tinyWinogrande** (acc_norm) | 46.48% | **48.56%** | 42.86% |

### Key Findings

- **NanoAgent-v0.2** achieves the best **instruction following** (IFEval) across all metrics (+5-8% improvement over v0.1)
- **NanoAgent-v0.1** leads on reasoning tasks: **tinyArc**, **tinyGSM8k**, and **tinyWinogrande**
- **NanoAgent-v0.2** improves on **tinyMMLU**, **tinyTruthfulQA**, and **tinyHellaswag** over both predecessors
- All NanoAgent versions significantly outperform the base SmolLM2-135M-Instruct on **IFEval** (instruction following)
- 🧰 **Tool Calling**: Only NanoAgent (v0.1 & v0.2) support tool calling — SmolLM2-135M-Instruct does not


## 🧭 Roadmap

- [ ] 📊 Benchmark more agentic tasks  
- [ ] 🧠 Explore GRPO for tool calling improvement  
- [ ] 🔀 Experiment with weight merging  
- [ ] 🧪 Evaluate multi-turn tool chaining  
- [ ] 🧹 Further refine datasets for stability


## Directory Tree

```
NanoAgent/
├── data/
│   ├── dataprep.py          # Dataset preparation, cleaning, and formatting
│   └── utils.py             # Helper utilities for data processing
│
├── grpo/
│   └── grpo-mlx.py          # Experimental GRPO (agentic fine-tuning) implementation using MLX
│
├── notebooks/
│   └── inference.ipynb      # Demo notebook for inference and evaluation
│
├── sft/
│   └── train-mlx.py         # Supervised Fine-Tuning (SFT) training script using MLX
│
├── utils/
│   ├── gguf_conv.py         # Conversion script for exporting model to GGUF format (for llama.cpp etc.)
│   ├── tokenizer.py         # Tokenizer helper functions and configs
│   └── webtool.py           # Example tool interface for web search / parsing integration
│
├── LICENSE                  # Apache 2.0 license file
├── NOTICE                   # Notices and attributions for datasets and dependencies
└── README.md                # Project overview, usage guide, and dataset details
```

---

## 📄 License

This project (code, model weights, and training recipes) is licensed under the [Apache License 2.0](./LICENSE).

## 📢 Notice

- Model & code are © [quwsarohi](https://github.com/QuwsarOhi), licensed under Apache 2.0.  
- Portions of the training data were sourced from third-party datasets under CDLA-P 2.0, MIT, CC-BY 4.0, ODC-BY, and Apache 2.0.  
- The licensors of these datasets do **not endorse** this project or its outputs.  
- If you redistribute or fine-tune this model, ensure your use complies with all applicable dataset licenses.


