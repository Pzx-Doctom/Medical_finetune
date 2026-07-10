# Ollama 模型部署完整指南

> 从 LoRA 微调项目代码到 Ollama 可运行模型的全流程文档。

---

## 目录

1. [前置总览：原始项目结构](#1-前置总览原始项目结构)
2. [阶段一：数据准备与 LoRA 微调](#2-阶段一数据准备与-lora-微调)
3. [阶段二：LoRA 合并到基座模型](#3-阶段二lora-合并到基座模型)
4. [阶段三：HuggingFace 格式转 GGUF 格式](#4-阶段三huggingface-格式转-gguf-格式)
5. [阶段四：导入 Ollama 并运行](#5-阶段四导入-ollama-并运行)
6. [完整流程图](#6-完整流程图)
7. [文件清单](#7-文件清单)
8. [常见问题排查](#8-常见问题排查)

---

## 1. 前置总览：原始项目结构

```
MedicalChatGPT/
├── config.py                  # 统一配置中心（基座模型、LoRA参数、训练超参）
├── train_lora.py              # LoRA 微调训练脚本
├── inference_compare.py       # 微调前后效果对比
├── deploy_local.py            # 原始本地部署（PEFT 加载方式）
├── data/
│   └── prepare_data.py        # 数据预处理（shibing624/medical → ChatML 格式）
├── results/
│   └── lora_adapter/          # ← 微调产物：LoRA 适配器权重（约 3MB）
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── tokenizer.json
└── requirements.txt
```

**关键信息**：

| 配置项 | 值 |
|--------|-----|
| 基座模型 | `Qwen/Qwen1.5-0.5B`（Qwen2 架构，24层，1024维） |
| 微调方法 | LoRA（r=8, alpha=16, target=q_proj+v_proj） |
| 训练数据 | `shibing624/medical`，采样 2000 条中文医疗问答 |
| 对话格式 | ChatML（`<|im_start|>system/user/assistant<|im_end|>`） |
| LoRA 产物 | `./results/lora_adapter/`（约 3MB，不能独立运行） |

---

## 2. 阶段一：数据准备与 LoRA 微调

### 这个阶段产出什么？

产出 `./results/lora_adapter/` 目录，包含 LoRA 增量权重。

### 执行的命令

```powershell
# 步骤 1: 数据预处理（将原始 dataset 转为 ChatML 格式的 JSONL）
python data/prepare_data.py
# → 输出: ./data/medical_qa_2000.jsonl

# 步骤 2: LoRA 微调训练
python train_lora.py
# → 输出: ./results/lora_adapter/（adapter_config.json + adapter_model.safetensors）
```

### 发生了什么？

```
shibing624/medical 数据集
  │  194 万条医疗问答
  │
  ▼  data/prepare_data.py（采样 2000 条 + ChatML 格式化）
  │
medical_qa_2000.jsonl
  │  每行格式：
  │  <|im_start|>system\n你是一个专业的医疗助手。<|im_end|>\n
  │  <|im_start|>user\n感冒了怎么办？<|im_end|>\n
  │  <|im_start|>assistant\n感冒是...<|im_end|>
  │
  ▼  train_lora.py（FP16 + gradient checkpointing）
  │
results/lora_adapter/
  ├── adapter_config.json      # LoRA 配置：r=8, alpha=16, 目标模块 q_proj/v_proj
  ├── adapter_model.safetensors # 增量权重（约 3MB）
  └── tokenizer.json           # tokenizer 配置
```

**核心原理**：LoRA 不修改原始模型权重，而是在 Attention 层的 `q_proj` 和 `v_proj` 旁注入两个低秩矩阵 A (d×r) 和 B (r×d)，只训练这两个小矩阵。因此产物只有约 3MB，不能独立运行，必须和基座模型配合使用。

---

## 3. 阶段二：LoRA 合并到基座模型

### 为什么需要合并？

Ollama 使用 GGUF 格式，**不支持直接加载 PEFT/LoRA adapter**。必须先将 LoRA 增量权重合并回基座模型，生成一个完整模型。

### 执行的命令

```powershell
python merge_lora.py
```

### 发生了什么？

```
results/lora_adapter/          Qwen/Qwen1.5-0.5B（从 HuggingFace 下载）
  │  adapter_model.safetensors   │  基座权重（~1.24GB）
  │  adapter_config.json         │  config.json
  │  (3MB)                       │  (1.24GB)
  │                              │
  └──────────┬──────────────────┘
             │  PEFT merge_and_unload()
             │  ΔW = A·B 合并到原始 Q、V 权重中
             ▼
       ./merged_model/
       ├── model.safetensors     # 合并后的完整权重（~1GB）
       ├── config.json           # 模型配置
       ├── tokenizer.json        # 分词器
       └── tokenizer_config.json
```

**关键代码**（`merge_lora.py` 核心逻辑）：

```python
# 1. 加载基座模型
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B", ...)

# 2. 加载 LoRA adapter
model = PeftModel.from_pretrained(base_model, "./results/lora_adapter")

# 3. 合并（核心步骤！）
model = model.merge_and_unload()
#   merge_and_unload 做了什么？
#   - 将 LoRA 的 A·B 矩阵乘积加到原始 Q/V 权重上
#   - 移除 PEFT 包装层，变回普通 Transformers 模型

# 4. 保存完整模型
model.save_pretrained("./merged_model")
```

**入出对比**：

| | 输入 | 输出 |
|---|------|------|
| 格式 | PEFT/LoRA adapter + 基座模型（分离） | 完整 HuggingFace 模型（合体） |
| 大小 | 3MB + 1.24GB | ~1GB |
| 能否独立运行 | ❌ 需要 PEFT 库加载 | ✅ 可直接用 transformers 加载 |

---

## 4. 阶段三：HuggingFace 格式转 GGUF 格式

### 为什么需要转换？

Ollama 底层使用 llama.cpp 推理引擎，只认 **GGUF**（GGML Unified Format）格式。HuggingFace 的 safetensors 格式无法直接给 Ollama 使用。

### 依赖安装

```powershell
# 克隆 llama.cpp（含转换工具）
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git ./llama.cpp

# 安装转换所需的 Python 依赖
pip install -r ./llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

# sentencepiece（Qwen 分词器需要）
pip install sentencepiece
```

### 执行的命令

```powershell
# F16 精度（无损，890MB）
python convert_to_gguf.py

# 或 Q4_K_M 量化（有损但极轻量，约 350MB）
python convert_to_gguf.py --q4_k_m
```

### 发生了什么？

```
./merged_model/                    llama.cpp/convert_hf_to_gguf.py
  │  model.safetensors              │  (llama.cpp 官方转换工具)
  │  config.json                    │
  │  tokenizer.json                 │
  │                                 │
  └────────────┬────────────────────┘
               │
               │  1. 读取 HF config，识别架构：Qwen2ForCausalLM
               │  2. 逐层转换权重（290 个 tensor）：
               │     - 大部分权重 float32 → F16（节省 50% 空间）
               │     - LayerNorm 等敏感层保持 F32
               │  3. 转换 tokenizer：SentencePiece → GGUF vocab
               │  4. 写入元数据：context_length=32768, 架构参数等
               │  5. 嵌入 ChatML 对话模板
               │
               ▼
  ./merged_model/medical-qwen1.5-0.5b-F16.gguf
  │  890.9 MB，单文件
  │  包含：模型权重 + tokenizer + 配置 + 对话模板
```

### 可选的量化精度

| 参数 | 精度 | 文件大小 | 速度 | 质量 | 适用场景 |
|------|------|---------|------|------|---------|
| `f16` | 半精度浮点 | 890 MB | 基准 | 无损 | 有充足磁盘空间 |
| `q8_0` | 8-bit 量化 | ~550 MB | 1.5x | 接近无损 | 通用部署 |
| `q4_k_m` | 4-bit K-quant | ~350 MB | 2x | 极小损失 | 本地低配机器 |

### 转换日志解读

```
INFO:hf-to-gguf:Model architecture: Qwen2ForCausalLM     ← 正确识别架构
INFO:hf-to-gguf:gguf: context length = 32768             ← 3.2 万 token 上下文
INFO:hf-to-gguf:gguf: embedding length = 1024            ← 隐藏维度
INFO:hf-to-gguf:gguf: head count = 16                    ← 注意力头数
INFO:gguf.vocab:Adding 151387 merge(s).                  ← BPE 词表合并规则
INFO:gguf.vocab:Setting chat_template to ...             ← ChatML 模板嵌入
```

---

## 5. 阶段四：导入 Ollama 并运行

### Modelfile 解析

```dockerfile
# 指定 GGUF 文件路径
FROM ./merged_model/medical-qwen1.5-0.5b-F16.gguf

# ChatML 对话模板（Go template 语法）
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

# 默认系统提示词
SYSTEM """你是一个专业的医疗助手。"""

# 生成参数
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 512

# 停止标记
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
```

**Modelfile 各字段作用**：

| 字段 | 作用 |
|------|------|
| `FROM` | 指向 GGUF 文件，Ollama 将模型权重导入 |
| `TEMPLATE` | 定义如何将用户消息 + 系统提示词组装成模型可理解的格式 |
| `SYSTEM` | 系统级提示词，定义模型角色行为 |
| `PARAMETER` | 推理超参：温度（随机性）、top_p（核采样）、最大生成长度 |
| `PARAMETER stop` | 遇到这些 token 时停止生成（防止模型"自问自答"） |

### 执行的命令

```powershell
# 导入模型（读取 GGUF + Modelfile，注册到 Ollama）
ollama create medical-assistant -f Modelfile

# 验证模型已注册
ollama list
# → 输出: medical-assistant:latest  ...  890 MB

# 交互式运行
ollama run medical-assistant

# 单次问答
ollama run medical-assistant "感冒了应该怎么办？"
```

### Ollama 导入过程发生了什么？

```
ollama create 命令
  │
  ├─ 1. gathering model components
  │     └─ 读取 Modelfile，解析 FROM、TEMPLATE、PARAMETER 等
  │
  ├─ 2. copying file sha256:xxx  100%
  │     └─ 将 GGUF 文件复制到 Ollama 模型库
  │        （默认路径：C:\Users\<用户名>\.ollama\models\）
  │
  ├─ 3. parsing GGUF
  │     └─ 解析 GGUF 文件结构，提取权重、tokenizer、元数据
  │
  ├─ 4. verifying conversion
  │     └─ 校验文件完整性
  │
  ├─ 5. creating new layer sha256:xxx
  │     └─ 为模型创建 Ollama 内部镜像层（类似 Docker layer）
  │
  └─ 6. writing manifest → success
        └─ 写入模型清单，注册完成
```

---

## 6. 完整流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    原始项目代码                                │
│                                                             │
│  shibing624/medical  ──→  data/prepare_data.py              │
│  (194万条医疗问答)         采样 2000 条，格式化为 ChatML       │
│                                                             │
│                          medical_qa_2000.jsonl               │
│                                                             │
│  Qwen/Qwen1.5-0.5B   ──→  train_lora.py                    │
│  (基座模型，1.24GB)        LoRA 微调（r=8, q_proj+v_proj）   │
│                                                             │
│                          results/lora_adapter/              │
│                          (LoRA 增量权重，3MB)                 │
└──────────────────────────────┬──────────────────────────────┘
                               │
          ╔════════════════════╧════════════════════╗
          ║          阶段二：合并 LoRA                ║
          ║                                        ║
          ║  python merge_lora.py                  ║
          ║  ├─ 加载基座模型 + LoRA adapter        ║
          ║  ├─ merge_and_unload() 合并            ║
          ║  └─ 保存完整模型                        ║
          ║                                        ║
          ║  ./merged_model/                       ║
          ║  ├─ model.safetensors  (~1GB)         ║
          ║  ├─ config.json                        ║
          ║  └─ tokenizer.json                     ║
          ╚════════════════════╤════════════════════╝
                               │
          ╔════════════════════╧════════════════════╗
          ║          阶段三：转 GGUF                 ║
          ║                                        ║
          ║  python convert_to_gguf.py             ║
          ║  ├─ 依赖：llama.cpp + sentencepiece    ║
          ║  ├─ 调用 convert_hf_to_gguf.py         ║
          ║  ├─ 逐层转换 290 个 tensor              ║
          ║  ├─ float32 → F16（省 50% 空间）       ║
          ║  └─ 嵌入 ChatML template               ║
          ║                                        ║
          ║  ./merged_model/                       ║
          ║    medical-qwen1.5-0.5b-F16.gguf       ║
          ║    (890.9 MB，单文件)                    ║
          ╚════════════════════╤════════════════════╝
                               │
          ╔════════════════════╧════════════════════╗
          ║          阶段四：Ollama 导入             ║
          ║                                        ║
          ║  ollama create medical-assistant       ║
          ║    -f Modelfile                        ║
          ║  ├─ 复制 GGUF 到模型库                  ║
          ║  ├─ 解析 + 校验                         ║
          ║  └─ 注册模型                            ║
          ║                                        ║
          ║  ollama run medical-assistant          ║
          ║  → 终端对话 / API 调用                  ║
          ╚══════════════════════════════════════════╝
```

### 文件格式转换链路

```
.safetensors (PEFT/LoRA)         [3MB, 不可独立运行]
        │  merge_and_unload()
        ▼
.safetensors (完整 HF 模型)       [~1GB, 可被 transformers 直接加载]
        │  convert_hf_to_gguf.py
        ▼
.gguf (llama.cpp 格式)           [890MB, 单文件包含权重+tokenizer+配置]
        │  ollama create
        ▼
Ollama 模型库 (.ollama/models/)  [Ollama 内部存储，可通过 ollama run 调用]
```

---

## 7. 文件清单

### 新增脚本（本次创建）

| 文件 | 阶段 | 输入 | 输出 | 作用 |
|------|------|------|------|------|
| `merge_lora.py` | 阶段二 | `results/lora_adapter/` + 基座模型 | `./merged_model/` | LoRA 合并为完整模型 |
| `convert_to_gguf.py` | 阶段三 | `./merged_model/` | `.gguf` 文件 | HF 格式转 GGUF |
| `Modelfile` | 阶段四 | `.gguf` 文件 | Ollama 注册 | Ollama 模型配置 |

### 原始项目文件（保持不变）

| 文件 | 作用 |
|------|------|
| `config.py` | 所有脚本共享的配置常量 |
| `data/prepare_data.py` | 数据预处理 |
| `train_lora.py` | LoRA 微调训练 |
| `inference_compare.py` | 微调前后对比 |
| `deploy_local.py` | 原始 PEFT 方式本地部署 |
| `results/lora_adapter/` | LoRA 微调产物（约 3MB） |

### 中间产物

| 目录/文件 | 说明 | 大小 |
|-----------|------|------|
| `./merged_model/` | 合并后的完整 HF 模型 | ~1GB |
| `./merged_model/medical-qwen1.5-0.5b-F16.gguf` | GGUF 格式模型 | 890.9MB |
| `./llama.cpp/` | llama.cpp 工具仓库 | 可删除 |

### 最终产物

| 名称 | 位置 | 访问方式 |
|------|------|---------|
| `medical-assistant` | Ollama 模型库 | `ollama run medical-assistant` |

---

## 8. 常见问题排查

### Q1：`git clone llama.cpp` 失败（代理问题）

```powershell
# 临时关闭代理再 clone
$env:HTTP_PROXY=''; $env:HTTPS_PROXY=''
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git ./llama.cpp
```

### Q2：`requirements-convert.txt not found`

新版 llama.cpp 已将文件改名为 `requirements-convert_hf_to_gguf.txt`：

```powershell
pip install -r ./llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
```

### Q3：`ModuleNotFoundError: No module named 'sentencepiece'`

```powershell
pip install sentencepiece
```

### Q4：`ollama create` 报 `no Modelfile found`

需要在项目根目录执行，或使用绝对路径：

```powershell
cd D:\Desktop\MedicalChatGPT
ollama create medical-assistant -f Modelfile
```

### Q5：想换个精度重新导入

```powershell
# 先删除旧模型
ollama rm medical-assistant

# 重新转换（如 Q4_K_M）
python convert_to_gguf.py --q4_k_m

# 修改 Modelfile 中的 FROM 路径指向新 .gguf 文件

# 重新导入
ollama create medical-assistant -f Modelfile
```

### Q6：模型的回答质量如何？

这是一个 **0.5B 参数**的小模型，仅用 2000 条数据微调了 3 个 epoch。期望管理：

- ✅ 能理解医疗场景的问答格式
- ✅ 回答风格偏向医疗助手角色
- ⚠️ 医学知识深度有限（模型太小）
- ⚠️ 复杂问题可能给出泛泛回答

如果需要更好的效果，可以：
1. 换更大的基座（如 Qwen2.5-7B）
2. 增加训练数据量
3. 使用 QLoRA 4-bit 量化训练降低成本

---

## 附录：常用 Ollama 命令

```powershell
ollama list                          # 列出所有已安装模型
ollama run medical-assistant         # 交互式对话
ollama rm medical-assistant          # 删除模型
ollama show medical-assistant        # 查看模型详情
ollama pull qwen2.5:0.5b            # 下载官方模型（对比用）
```

### Python API 调用

```python
import ollama

response = ollama.chat(
    model="medical-assistant",
    messages=[{"role": "user", "content": "感冒了应该怎么办？"}]
)
print(response["message"]["content"])
```

### HTTP API 调用

```powershell
Invoke-RestMethod -Uri "http://localhost:11434/api/chat" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"model":"medical-assistant","messages":[{"role":"user","content":"感冒了怎么办？"}],"stream":false}'
```
