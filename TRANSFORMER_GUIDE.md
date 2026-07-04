# Transformer 从零入门 — 配套医疗 LoRA 微调项目

> 本文档为没有 Transformer 基础的读者编写，从"为什么需要它"讲起，逐步拆解架构，
> 并与你实际跑通的医疗 LoRA 微调项目对应，帮助你理解代码里的每个概念。

---

## 一、为什么需要 Transformer

### 1.1 早期怎么处理文本

在 Transformer 之前，处理文本主要用 RNN（循环神经网络）和 LSTM。

**RNN 的工作方式**：逐词阅读，像人读书一样一个词一个词看下去。

```
输入："感冒 发烧 咳嗽"
RNN:  看"感冒" → 记住 → 看"发烧" → 更新记忆 → 看"咳嗽" → 更新记忆 → 输出
```

**RNN 的问题**：
1. **记不住远处的词**：读到"咳嗽"时，"感冒"的信息已经模糊了（叫"长距离依赖问题"）
2. **无法并行**：必须读完第 1 个词才能读第 2 个，训练慢

### 1.2 Transformer 怎么解决

2017 年 Google 发表《Attention Is All You Need》，提出 Transformer，核心创新：

1. **一次性看所有词**：不再逐词读，而是同时看整句话，每个词都能直接"看到"其他所有词
2. **完全可并行**：所有词同时处理，训练速度快几个数量级

这就是今天所有大模型（GPT、Qwen、LLaMA）的基础架构。

---

## 二、核心概念：注意力机制（Self-Attention）

### 2.1 用一个比喻理解

假设你在读这句话："**感冒**会导致**发烧**和**咳嗽**"。

当你读到"咳嗽"这个词时，你的大脑会自动关联到"感冒"——因为"咳嗽"是"感冒"的症状。这种"自动关联相关词"的能力，就是**注意力机制**。

### 2.2 注意力的三个角色：Q、K、V

借用"信息检索"的思路，每个词会被变换成三种角色：

| 角色 | 全称 | 含义 | 比喻 |
|------|------|------|------|
| **Q** | Query | "我在找什么" | 搜索关键词 |
| **K** | Key | "我有什么标签" | 书的标题/标签 |
| **V** | Value | "我的实际内容" | 书的正文 |

**计算流程**（以"咳嗽"这个词为例）：

```
1. "咳嗽"生成自己的 Q → "我在找：导致我的原因"
2. 句子中每个词都亮出自己的 K：
   - "感冒".K → "我是疾病"
   - "发烧".K → "我是症状"
   - "咳嗽".K → "我是症状"
3. "咳嗽".Q 和每个 .K 匹配，算出关注程度：
   - 关注"感冒" 80%（因为咳嗽是感冒的症状）
   - 关注"发烧" 15%
   - 关注"自己" 5%
4. 按关注程度，把对应的 V 加权求和：
   → "咳嗽"的新表示 = 0.8×感冒.V + 0.15×发烧.V + 0.05×咳嗽.V
```

这样，"咳嗽"这个词的表示就融入了"感冒"的信息——模型理解了它们的关联。

### 2.3 数学公式（简化版）

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d) · V
```

拆解：
1. `Q · Kᵀ`：算每个词和其他词的关联度（点积越大越相关）
2. `/ √d`：缩放，防止数值太大（d 是维度）
3. `softmax`：把关联度变成概率（加起来等于 1）
4. `· V`：按概率加权求和，得到最终输出

### 2.4 这就是"自注意力"（Self-Attention）

为什么叫"自"？因为 Q、K、V 都来自**同一个输入**——句子自己和自己做匹配。不是去外部数据库检索，而是句子内部的词互相"看"彼此。

---

## 三、Q、K、V 怎么来的：投影矩阵

### 3.1 输入不能直接当 Q/K/V

输入的词向量（embedding）是"原始表示"，不能直接当 Q/K/V 用。需要先用矩阵**投影/变换**：

```
Q = X · W_q    （W_q 就是 q_proj）
K = X · W_k    （W_k 就是 k_proj）
V = X · W_v    （W_v 就是 v_proj）
```

- `X`：输入（所有词的向量，形状 `[句子长度, 维度]`）
- `W_q / W_k / W_v`：可学习的权重矩阵（就是代码里的 `q_proj / k_proj / v_proj`）
- `Q / K / V`：投影后的结果

### 3.2 投影矩阵的作用

**打个比方**：你有一批原始数据（X），要放进检索系统。不能直接放，要先"格式化"：
- `W_q`：把数据格式化成"查询语句"（Q）
- `W_k`：把数据格式化成"索引标签"（K）
- `W_v`：把数据格式化成"可检索内容"（V）

这三个矩阵是**可学习的**——训练时模型会调整它们，让 Q/K/V 越来越好用。

### 3.3 对应到你的项目

你的 `config.py` 里：

```python
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
```

这就是告诉 LoRA："在这两个投影矩阵上注入低秩适配器"。选择 q_proj 和 v_proj 的原因见 [第六章](#六为什么-lora-选-q_proj-和-v_proj)。

---

## 四、Transformer 的完整架构

### 4.1 整体结构（以 Qwen1.5 为例）

```
输入文本 "感冒怎么办"
    ↓
Token化 → [词1, 词2, 词3]          （Tokenizer）
    ↓
词向量嵌入 → [向量1, 向量2, 向量3]   （Embedding 层）
    ↓
┌─────────────────────────┐
│  Transformer Block × N  │  ← Qwen1.5-0.5B 有 24 层
│  ┌───────────────────┐  │
│  │ 1. 自注意力层      │  │  ← q_proj/k_proj/v_proj 在这里
│  │ 2. 残差连接 + 归一化│  │
│  │ 3. 前馈网络 (FFN)  │  │  ← MLP 层（gate_proj/up_proj/down_proj）
│  │ 4. 残差连接 + 归一化│  │
│  └───────────────────┘  │
└─────────────────────────┘
    ↓
输出层 → 预测下一个词          （LM Head）
```

### 4.2 每一层做了什么

**Transformer Block** 是重复堆叠的基本单元，Qwen1.5-0.5B 堆了 24 层。每层做两件事：

#### 第一步：自注意力（让词互相"看到"彼此）

```
输入: [词1, 词2, 词3]
  ↓ 用 q_proj/k_proj/v_proj 算出 Q/K/V
  ↓ Attention 计算
输出: [词1', 词2', 词3']  ← 每个词都融入了其他词的信息
```

#### 第二步：前馈网络 FFN（对每个词独立做变换）

```
输入: [词1', 词2', 词3']
  ↓ 对每个词单独过一个两层的全连接网络
  ↓ gate_proj → 激活函数 → down_proj
输出: [词1'', 词2'', 词3'']  ← 每个词的信息被进一步加工
```

**FFN 的作用**：注意力层负责"词与词的信息融合"，FFN 负责"单个词的信息加工"。两者交替进行。

#### 残差连接 + LayerNorm

每一步前后都有：
```
输出 = LayerNorm(输入 + 子层(输入))
```

- **残差连接**（`输入 + 子层(输入)`）：防止深层网络梯度消失，让信息能"直达"
- **LayerNorm**：把数值归一化到稳定范围，防止数值爆炸/消失

### 4.3 多头注意力（Multi-Head Attention）

实际中，注意力不是只做一次，而是做多次，每次叫一个"头"：

```
头1：关注"症状关系"（感冒→发烧）
头2：关注"语法关系"（主语→谓语）
头3：关注"位置关系"（前后词）
...
```

多个头的结果拼接起来，再过一个 `o_proj`（输出投影）整合。

**Qwen1.5-0.5B 的注意力有 16 个头**，每个头独立学习不同的关注模式。

---

## 五、Decoder-only 架构（Qwen/GPT 的选择）

### 5.1 三种 Transformer 架构

| 架构 | 代表模型 | 特点 | 用途 |
|------|---------|------|------|
| Encoder-only | BERT | 看全部词，双向理解 | 文本分类、NER |
| Decoder-only | GPT、Qwen | 只看前面的词，单向生成 | 文本生成（你的项目） |
| Encoder-Decoder | T5、BART | 编码+解码 | 翻译、摘要 |

### 5.2 Decoder-only 的关键：因果掩码（Causal Mask）

生成文本时，模型只能看"已经出现"的词，不能偷看"未来"的词：

```
生成"感冒怎么办"时：
- 预测"感"：只能看到 <开头>
- 预测"冒"：只能看到 <感>
- 预测"怎"：只能看到 <感冒>
- ...
```

实现方式：在注意力计算时，用一个**下三角掩码**把"未来"的注意力权重设为 -∞（softmax 后变 0）。

### 5.3 对应到你的训练

你的训练数据格式：
```
<|im_start|>system
你是一个医疗助手。<|im_end|>
<|im_start|>user
感冒怎么办？<|im_end|>
<|im_start|>assistant
多休息，多喝水。<|im_end|>
```

模型的任务是**逐词预测下一个 token**：
- 看到 system + user → 预测 assistant 的第一个词
- 看到 system + user + "多" → 预测 "休息"
- 以此类推

配合 **label masking**：只在 assistant 部分计算 loss，不计算 system/user 部分的 loss。

---

## 六、为什么 LoRA 选 q_proj 和 v_proj

### 6.1 四个投影矩阵的作用回顾

| 矩阵 | 作用 | 微调它影响什么 |
|------|------|--------------|
| **q_proj** | 算 Query（我在找什么） | 改变"关注什么" |
| k_proj | 算 Key（我的标签） | 改变"被别人关注的条件" |
| **v_proj** | 算 Value（我的内容） | 改变"提供什么信息" |
| o_proj | 输出整合 | 改变"如何整合多头结果" |

### 6.2 为什么是 q_proj + v_proj

LoRA 原始论文的实验结论：

1. **q_proj 最关键**：决定"关注什么"。医疗微调的核心是让模型学会"关注医疗相关信息"，微调 q_proj 最直接有效。

2. **v_proj 次关键**：决定"提取什么内容"。微调 v_proj 让模型学会"提取医疗知识"。

3. **k_proj 和 o_proj 收益小**：实验表明加它们效果提升不明显，但参数量增加。性价比低。

### 6.3 如果效果不好可以扩展

```python
# 保守配置（你的项目）：参数最少
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# 标准配置：效果更好，参数翻倍
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# 激进配置：效果最好，参数最多
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
```

你的项目用保守配置，参数量 78 万（0.17%），是验证 LoRA 原理的合理选择。

---

## 七、Transformer 中所有可被 LoRA 注入的位置

以 Qwen1.5-0.5B 为例，每一层 Transformer Block 的可训练矩阵：

```
Transformer Block（× 24 层）
│
├── 自注意力层
│   ├── q_proj  [1024, 1024]   ← 你注入了 LoRA ✅
│   ├── k_proj  [1024, 1024]   ← 未注入
│   ├── v_proj  [1024, 1024]   ← 你注入了 LoRA ✅
│   └── o_proj  [1024, 1024]   ← 未注入
│
└── 前馈网络 FFN
    ├── gate_proj [1024, 2816]  ← 未注入
    ├── up_proj   [1024, 2816]  ← 未注入
    └── down_proj [2816, 1024]  ← 未注入
```

**你的 LoRA 配置**：
- 注入：q_proj + v_proj（每层 2 个矩阵 × 24 层 = 48 个矩阵）
- 每个 LoRA 增量：`[1024, 8] + [8, 1024]` = 16,384 参数
- 总可训练参数：48 × 16,384 = 786,432 ≈ 78 万 ✅（和训练日志一致）

---

## 八、从 Transformer 到你的训练代码

### 8.1 训练时发生了什么

```python
# 1. 输入文本 → Token → 向量
inputs = tokenizer("感冒怎么办？")  # → [词1_id, 词2_id, 词3_id]
embeddings = embedding_layer(inputs)  # → [向量1, 向量2, 向量3]

# 2. 经过 24 层 Transformer Block
for block in transformer.blocks:  # 24 层
    # 2.1 自注意力
    Q = embeddings @ block.q_proj  # ← LoRA 在这里生效：W_q + A·B
    K = embeddings @ block.k_proj
    V = embeddings @ block.v_proj  # ← LoRA 在这里生效：W_v + A·B
    attn_out = attention(Q, K, V)
    attn_out = attn_out @ block.o_proj

    # 2.2 前馈网络
    ff_out = block.down_proj(activation(block.gate_proj(embeddings)
                                       * block.up_proj(embeddings)))

    # 2.3 残差 + LayerNorm
    embeddings = layernorm(embeddings + attn_out + ff_out)

# 3. 输出预测
logits = lm_head(embeddings)  # → 每个位置预测下一个词的概率
loss = cross_entropy(logits, labels)  # labels 中 -100 的位置不计算
```

### 8.2 LoRA 在哪里生效

```python
# 原始 q_proj：只有 W_q
Q = X @ W_q

# LoRA 注入后：W_q 冻结，额外加 A·B
Q = X @ W_q + X @ (A @ B)
#         ↑冻结      ↑可训练（A 和 B 是 LoRA 参数）
```

- `W_q`：464,774,144 - 786,432 ≈ 4.64 亿参数，**冻结不训练**
- `A·B`：786,432 参数，**只训练这些**

这就是你训练日志里 `trainable%: 0.1692%` 的来源。

---

## 九、关键术语速查表

| 术语 | 英文 | 通俗解释 |
|------|------|---------|
| Token | Token | 文本的最小单位（一个字或词） |
| Embedding | 词嵌入 | 把 token 变成向量 |
| Self-Attention | 自注意力 | 句子内的词互相"看"彼此 |
| Q/K/V | Query/Key/Value | 注意力的三个角色（搜索/标签/内容） |
| q_proj/k_proj/v_proj | 投影矩阵 | 把输入变换成 Q/K/V 的权重矩阵 |
| Multi-Head | 多头注意力 | 多个注意力并行，关注不同模式 |
| FFN | 前馈网络 | 对单个词的信息做加工的两层全连接 |
| LayerNorm | 层归一化 | 把数值稳定在合理范围 |
| Residual | 残差连接 | 输入直接加到输出，防止梯度消失 |
| Causal Mask | 因果掩码 | 防止看到"未来"的词 |
| Decoder-only | 纯解码器 | 只做生成，不看未来（GPT/Qwen 的架构） |
| LM Head | 语言模型头 | 把向量变成词的概率分布 |
| Label Masking | 标签遮蔽 | prompt 部分不算 loss，只算回答部分 |
| LoRA | 低秩适配 | 冻结原权重，只训练小矩阵 A·B |

---

## 十、进一步学习建议

### 想深入理解 Transformer

1. **必读论文**：《Attention Is All You Need》（2017，Transformer 原始论文）
2. **可视化推荐**：Jay Alammar 的《The Illustrated Transformer》（图文讲解，搜得到中文版）
3. **视频推荐**：3Blue1Brown 的 Transformer 系列可视化视频

### 想深入理解 LoRA

1. **必读论文**：《LoRA: Low-Rank Adaptation of Large Language Models》（2021，LoRA 原始论文）
2. **实践**：对照你的 `train_lora.py` 代码，逐行理解每个配置的作用

### 想深入理解你的项目

1. 读 `config.py`：理解每个参数的含义
2. 读 `train_lora.py`：对照本文档第八章，理解训练流程
3. 读 `inference_compare.py`：理解推理时如何加载 LoRA adapter

---

## 一句话总结

> Transformer 的核心是"自注意力机制"——让句子里的每个词都能直接看到其他所有词，通过 Q（我在找什么）、K（我有什么标签）、V（我的内容）三个角色实现信息融合。q_proj 和 v_proj 就是把输入变换成 Q 和 V 的权重矩阵。你的 LoRA 项目选择只微调这两个矩阵，因为它们最影响"关注什么"和"提取什么"，用最少的参数实现最大的领域适配效果。
