

# Build a Large Language Model (From Scratch) 

by Sebastian Raschka



Updated 1606 GMT+8 Feb 21 2025

2025 spring, Complied by Hongfei Yan





# 第一章 理解大语言模型

Understanding Large Language Models

近年来，大语言模型（LLM）如 OpenAI 的 ChatGPT 取得了重大突破，开启了自然语言处理（NLP）的新纪元。在 LLM 出现之前，传统的 NLP 方法在 **电子邮件垃圾分类** 或 **简单模式识别** 方面表现良好，但在 **复杂语言理解** 和 **文本生成** 方面仍有很大局限。例如，以前的语言模型难以根据关键词列表生成一封完整的电子邮件，而现代 LLM 可以轻松完成这一任务。

---

## 1.1 什么是 LLM？

LLM（大语言模型）是一种神经网络，旨在 **理解、生成和响应类人文本**。它们使用 **深度学习** 进行训练，并在 **海量文本数据** 上进行大规模优化。

### 1.1.1 “大”语言模型的含义
“**大（Large）**”不仅指模型参数的规模，还指其训练所用的海量数据集。例如：
- GPT-3 拥有 **1750 亿个参数**。
- LLaMA 2 训练数据包含 **StackExchange 编程问答（78GB）和 ArXiv 论文（92GB）**。

模型的核心训练目标是 **下一个词预测（Next-Word Prediction）**：
$$
P(w_t | w_1, w_2, ..., w_{t-1})
$$
尽管这一任务看似简单，但它能训练出极为强大的模型，使其具备 **文本理解、语境推理** 甚至 **创意写作** 的能力。

---

## 1.2 LLM 的应用

LLM 具有广泛的应用场景，例如：
- **机器翻译**（如 DeepL、Google Translate）
- **文本摘要**（如新闻摘要、法律文档摘要）
- **情感分析**（如分析社交媒体情绪）
- **代码生成**（如 GitHub Copilot）
- **智能助手**（如 ChatGPT、Google Gemini）。

此外，LLM 在 **专业领域** 也发挥了重要作用，如：
- **法律分析**（法律合同审查）
- **医学诊断**（分析病历、提供医学建议）
- **金融分析**（市场预测、风险评估）。

---

## 1.3 LLM 的构建与使用流程

LLM 的构建通常分为以下三个阶段：
1. **预训练（Pretraining）**：使用海量无标签文本数据训练基础模型。
2. **微调（Fine-Tuning）**：在特定任务或领域数据集上调整模型权重。
3. **推理（Inference）**：在实际应用中运行模型，如文本生成、问答。

为了优化训练效果，研究者通常采用 **自监督学习（Self-Supervised Learning）**，即：
- **不需要人工标注数据**，而是使用文本本身作为监督信号。
- 例如，GPT 通过 **预测下一个单词** 进行训练，使其能够理解文本结构。

---

## 1.4 Transformer 结构简介

现代 LLM 主要基于 **Transformer** 结构，该架构由 Google 在 2017 年提出（论文 *Attention Is All You Need*）。Transformer 的核心组件包括：
- **自注意力（Self-Attention）**：使模型能够关注句子中不同单词的关系。
- **前馈神经网络（Feed Forward Network, FFN）**：对注意力层的输出进行进一步处理。
- **残差连接（Residual Connections）** 和 **层归一化（Layer Normalization）**：提高训练稳定性。

### 1.4.1 编码器-解码器架构
Transformer 由 **编码器（Encoder）** 和 **解码器（Decoder）** 组成：
- **BERT** 使用 **编码器**，适用于 **文本分类**、**情感分析** 等任务。
- **GPT** 使用 **解码器**，适用于 **文本生成** 任务。

GPT 的 **解码器结构** 采用 **单向注意力（Unidirectional Attention）**，即：
$$
\text{每个单词只能关注之前的单词，而不能看到未来的单词}
$$
这种设计使 GPT 能够进行 **自回归文本生成（Autoregressive Generation）**。

---

## 1.5 预训练所需的大规模数据集

训练 LLM 需要 **极其庞大的数据集**。例如：
| 数据集      | 描述         | 规模                   |
| ----------- | ------------ | ---------------------- |
| CommonCrawl | 网络爬取文本 | 4100 亿个标记（570GB） |
| WebText2    | 网络文章     | 190 亿个标记           |
| BooksCorpus | 电子书数据集 | 12 亿个标记            |
| Wikipedia   | 维基百科数据 | 30 亿个标记            |

GPT-3 的预训练数据集总规模 **超过 3000 亿个标记**，训练成本 **高达 460 万美元**。

---

## 1.6 GPT 结构解析

### 1.6.1 GPT 训练任务
GPT 采用 **下一个词预测（Next-Word Prediction）** 任务，即：
```
输入："The sky is"
GPT 预测："blue"
```
这是一种 **自监督学习**，模型利用已有数据自动生成训练样本。

### 1.6.2 GPT 架构
GPT 仅使用 **Transformer 解码器（Decoder-Only Architecture）**，其结构包括：
- **输入嵌入（Token Embedding）**：将文本转换为向量。
- **位置编码（Positional Encoding）**：为 Transformer 添加位置信息。
- **多层 Transformer 解码器**：
  - **多头自注意力（Multi-Head Self-Attention）**
  - **前馈神经网络（FFN）**
  - **残差连接（Residual Connection）**
  - **层归一化（LayerNorm）**
- **输出层**：预测下一个单词。

这种架构的优点：
✅ **支持长文本上下文**
✅ **高效的并行计算**
✅ **强大的语言建模能力**

> Q. ChatGPT的Transformer模型通能够有效处理的序列长度大约是多少？
>
> Transformer 模型能够有效处理的 **序列长度（context length）** 取决于 **架构设计、计算资源和优化方法**。一般来说：
>
> 1. **标准 Transformer（如 GPT-2、BERT）**
>    - 典型最大序列长度：**512 - 2048 tokens**
>    - 主要受 **自注意力（Self-Attention）计算复杂度 O(n²)** 限制。
>
> 2. **GPT-3**
>    - 最大序列长度：**2048 tokens**
>    - **计算开销**：随着长度增加，计算资源需求大幅增长。
>
> 3. **GPT-4 和 GPT-4 Turbo**
>    - 最大序列长度：**128k tokens**（部分版本支持更长）。
>
> 4. **Longformer / BigBird（优化长序列）**
>    - **窗口化注意力（Windowed Attention）** 降低计算需求。
>    - 可处理 **8k - 32k tokens**，甚至更长。
>
> 5. **Mistral / Gemini / Claude 3**
>    - 采用更高效的 **注意力机制（如稀疏注意力）**，支持 **长达 200k tokens** 的上下文。
>
> **你的 Transformer 能处理的序列长度**
>
> 如果你是 **从零实现 GPT**，受限于标准 **O(n²) 自注意力计算**：
> - **推荐最大长度：512 - 2048 tokens**
> - **如果 GPU 内存足够，可尝试 4096 tokens**
> - **优化方法**：
>   - **稀疏注意力（Sparse Attention）**
>   - **线性 Transformer（如 Performer, Linformer）**
>   - **分块计算（Sliding Window Attention）**
>
> 



---

## 1.7 如何从零构建 LLM？

本书的核心目标是 **从零实现一个类似 ChatGPT 的 LLM**。我们将按照以下步骤进行：
1. **数据准备与分词（Tokenization）**
2. **实现注意力机制（Self-Attention）**
3. **构建完整的 Transformer 解码器**
4. **进行预训练（Pretraining）**
5. **模型微调（Fine-Tuning）**
6. **评估 LLM 生成能力**
7. **加载预训练权重**
8. **优化 LLM 以执行特定任务（如分类、对话）**。

---

## 本章总结

- **LLM 通过 Transformer 结构进行训练，具有强大的语言理解和生成能力**。
- **GPT 采用解码器架构，专注于自回归文本生成**。
- **训练 LLM 需要大规模数据集，如 CommonCrawl 和 Wikipedia**。
- **本书将详细讲解 LLM 的实现，包括分词、注意力机制、预训练和微调**。

在下一章，我们将深入探讨 **文本数据处理**，学习如何 **分词、构建词汇表**，并将文本转换为 **可输入 LLM 的数值表示**！





# 第二章 处理文本数据

Working with Text Data

在第一章中，我们介绍了 **大语言模型（LLM）** 的基本结构，并了解了它们是如何在 **大规模文本数据** 上进行预训练的。本章的目标是学习 **如何准备 LLM 训练所需的文本数据**，包括：
- **文本分词（Tokenization）**
- **子词级别分割（Subword Tokenization）**
- **字节对编码（Byte Pair Encoding, BPE）**
- **滑动窗口采样（Sliding Window Sampling）**
- **将标记转换为 LLM 输入向量**。

---

## 2.1 词嵌入（Word Embeddings）

神经网络无法直接处理原始文本。文本是 **类别数据（categorical data）**，不适用于矩阵计算，因此我们需要一种方法将文本转换为 **连续值向量**（Continuous-valued vectors），即 **词嵌入（Word Embeddings）**。

### 2.1.1 词嵌入的作用
词嵌入的目标是将 **离散单词映射到连续向量空间**，这样：
- **相似单词的向量距离更近**（如“猫”和“狗”）。
- **能够捕捉语法和语义信息**（如动词和名词的区别）。
- **可以作为 LLM 的输入，提高文本建模能力**。

不同的数据类型需要不同的嵌入方法，例如：
| 数据类型      | 嵌入方式                                  |
| ------------- | ----------------------------------------- |
| 文本（Text）  | 词嵌入模型（如 Word2Vec, GPT Embeddings） |
| 音频（Audio） | 音频嵌入模型                              |
| 视频（Video） | 视频嵌入模型                              |

---

## 2.2 文本分词（Tokenizing Text）

### 2.2.1 什么是分词？
分词（Tokenization）是将文本拆分为**单词（word）、子词（subword）或字符（character）**的过程。不同的分词方式适用于不同的任务。

例如：
```python
import re
text = "Hello, world. This is a test."
tokens = re.split(r'(\s)', text)
print(tokens)
```
**输出**：
```
['Hello,', ' ', 'world.', ' ', 'This', ' ', 'is', ' ', 'a', ' ', 'test.']
```
这种方式虽然能基本拆分单词，但仍然包含 **标点符号**，需要进一步优化。

### 2.2.2 改进的分词方法
为了更准确地分割文本，我们可以：
1. **保留标点符号作为独立标记**
2. **去除不必要的空格**
3. **保持大小写信息**

优化后的代码：
```python
tokens = re.split(r'([,.]|\s)', text)
tokens = [t.strip() for t in tokens if t.strip()]
print(tokens)
```
**输出**：
```
['Hello', ',', 'world', '.', 'This', 'is', 'a', 'test', '.']
```
现在，每个单词和标点符号都被正确拆分成独立的标记。

---

## 2.3 将标记转换为 Token ID

分词后，我们需要将 **标记映射为整数索引（Token IDs）**，以便 LLM 处理。我们可以使用 **词汇表（Vocabulary）** 将每个单词映射为唯一 ID。

### 2.3.1 创建词汇表
```python
vocab = {"Hello": 0, ",": 1, "world": 2, ".": 3, "This": 4, "is": 5, "a": 6, "test": 7}
tokens = ['Hello', ',', 'world', '.', 'This', 'is', 'a', 'test', '.']
token_ids = [vocab[t] for t in tokens]
print(token_ids)
```
**输出**：
```
[0, 1, 2, 3, 4, 5, 6, 7, 3]
```
这样，我们就把文本转换为 LLM 可处理的数字序列。

---

## 2.4 添加特殊标记（Special Tokens）

为了帮助 LLM 处理不同的任务，我们通常会 **在文本中添加特殊标记**，例如：
- **[BOS]**（Beginning of Sequence）——标记文本开头
- **[EOS]**（End of Sequence）——标记文本结尾
- **[PAD]**（Padding Token）——用于填充短文本
- **[UNK]**（Unknown Token）——表示未见过的单词

```python
special_tokens = {"[BOS]": 100, "[EOS]": 101, "[PAD]": 102, "[UNK]": 103}
```
例如，我们可以使用 **[EOS]** 标记多个文本段落的边界：
```
"Hello, world. <|endoftext|> This is a test. <|endoftext|>"
```
这样，LLM 在训练时可以学习不同文本之间的分界。

---

## 2.5 字节对编码（Byte Pair Encoding, BPE）

LLM 采用 **子词级别的分词**，而不是简单的单词分词。**字节对编码（BPE）** 是一种流行的方法，它通过 **合并最常见的字符或子词对** 逐步构建更大的标记。

### 2.5.1 BPE 示例
假设我们有单词 "low", "lower", "newest", "widest"，BPE 的合并过程如下：
1. 统计字符频率：
   ```
   {'l': 1, 'o': 1, 'w': 1, 'e': 3, 'r': 2, 'n': 1, 's': 2, 't': 2, 'd': 1, 'i': 1}
   ```
2. 合并频率最高的字符对（如 'e' 和 's'），直到满足词汇表大小限制。

BPE 使 LLM 能够更好地处理：
- 罕见单词（分解成更小的子词）
- 词缀（例如“un-” 或 “-ing”）
- 新造词（如“deepfake”）。

---

## 2.6 滑动窗口采样（Sliding Window Sampling）

LLM 训练时，每个输入文本块都有固定的 **上下文窗口（Context Window）**，例如 128 或 512 个 token。为了 **覆盖整个数据集**，我们采用 **滑动窗口** 采样方法。

### 2.6.1 滑动窗口示例
假设我们有以下文本：
```
"The old library stood in the center of town."
```
如果窗口大小为 4，我们可以生成：
```
[The, old, library, stood] -> 目标: in
[old, library, stood, in] -> 目标: the
[library, stood, in, the] -> 目标: center
...
```
这样，LLM 可以 **学习整个句子的上下文**。

---

## 2.7 词向量嵌入（Token Embeddings）

最终，我们需要将 **Token ID 转换为向量** 作为 LLM 的输入。通常，LLM 使用 **可训练的嵌入层（Embedding Layer）**：
```python
import torch.nn as nn
embedding_layer = nn.Embedding(num_embeddings=50000, embedding_dim=768)
token_ids = torch.tensor([0, 2, 5, 10])  # 示例 Token ID
embeddings = embedding_layer(token_ids)
```
LLM 会在训练过程中**优化这些向量**，以便更好地捕捉文本的语义关系。

---

## 本章总结

- **文本数据处理** 是 LLM 训练的关键前置步骤。
- **BPE 分词** 可以高效地处理未知单词和罕见词。
- **滑动窗口采样** 确保训练数据的覆盖范围更广。
- **Token 嵌入** 将 Token ID 映射到高维向量，使 LLM 能够学习语义信息 **注意力机制（Attention Mechanisms）**，这是 Transformer 和 GPT 的核心组件！ 



# 第三章 编码注意力机制

Coding Attention Mechanisms

在本章中，将探讨 LLM（大语言模型）架构的一个核心组成部分——**注意力机制**。首先单独分析注意力机制的工作原理，然后编写代码实现 LLM 的其余部分，使其能够生成文本。

## 本章内容：
- 在神经网络中使用注意力机制的原因
- 从基础的自注意力框架逐步增强到更高级的注意力机制
- 允许 LLM 逐个生成标记的因果注意力模块
- 通过丢弃随机选择的注意力权重来减少过拟合
- 将多个因果注意力模块堆叠成多头注意力模块

---

## 3.1 处理长序列的问题

在 LLM 之前的架构中，不包括注意力机制的模型难以处理长序列。例如，在机器翻译任务中，我们无法简单地逐词翻译文本，因为不同语言的语法结构不同。

解决这个问题的常见方法是使用 **编码器-解码器（encoder-decoder）** 深度神经网络。编码器首先读取并处理整个输入文本，而解码器则生成翻译后的文本。

---

## 3.2 通过注意力机制捕捉数据依赖关系

早期的 RNN（循环神经网络）架构在处理较长文本时存在问题，因为它们无法直接访问较早的隐藏状态，而只能依赖当前隐藏状态来编码所有相关信息。这可能会导致上下文丢失，特别是在长句子中，单词之间的依赖关系可能跨越较长的距离。

因此，2014 年提出了 **Bahdanau 注意力机制**，允许解码器在生成每个标记时有选择地访问输入序列的不同部分。随后，研究人员发现可以完全放弃 RNN，而是直接采用 **Transformer 结构**，其核心机制正是 **自注意力（Self-Attention）**。

---

## 3.3 使用自注意力关注输入的不同部分

### 3.3.1 无可训练权重的简化自注意力机制

首先实现一个简化的自注意力机制，该机制不涉及可训练的权重。其目标是先理解自注意力的核心概念，然后再引入可训练参数。

在自注意力机制中，每个输入位置都会考虑输入序列中所有其他位置的相关性。具体来说，计算一个 **上下文向量（context vector）**，它是输入序列所有元素的加权和，其中权重由 **注意力权重（attention weights）** 确定。

---

## 3.4 具有可训练权重的自注意力实现

### 3.4.1 逐步计算注意力权重
使用以下步骤计算 **缩放点积注意力（Scaled Dot-Product Attention）**：
1. 计算 **查询（Query）、键（Key）、值（Value）**：
   $$
   Q = X W_Q, \quad K = X W_K, \quad V = X W_V
   $$
   
2. 计算注意力分数：
   $$
   A = \frac{Q K^T}{\sqrt{d_k}}
   $$
   
3. 通过 Softmax 归一化：
   $$
   \alpha = \text{Softmax}(A)
   $$
   
4. 计算最终的上下文向量：
   $$
   Z = \alpha V
   $$
   

### 3.4.2 编写紧凑的自注意力 Python 类

```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec
```

---

## 3.5 使用因果注意力隐藏未来单词

### 3.5.1 应用因果注意力掩码
在 LLM 任务中，希望模型在预测下一个标记时，仅考虑当前位置及其之前的标记，而不允许其访问未来的信息。为此，我们使用 **因果注意力（Causal Attention）**，它通过掩码（masking）确保模型只能看到当前及之前的输入。

可以通过上三角矩阵来实现掩码：
```python
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked_scores = attn_scores.masked_fill(mask.bool(), -torch.inf)
```

### 3.5.2 使用 Dropout 掩码额外的注意力权重
在 Transformer 结构（如 GPT）中，通常在计算完注意力权重后，应用 **Dropout** 以减少过拟合：
```python
dropout = nn.Dropout(0.1)
attn_weights = dropout(attn_weights)
```

### 3.5.3 实现紧凑的因果注意力类
```python
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout_rate=0.1):
        super().__init__()
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        mask = torch.triu(torch.ones_like(attn_scores), diagonal=1)
        masked_scores = attn_scores.masked_fill(mask.bool(), -torch.inf)
        attn_weights = self.dropout(torch.softmax(masked_scores / keys.shape[-1]**0.5, dim=-1))
        context_vec = attn_weights @ values
        return context_vec
```

---

## 3.6 从单头扩展到多头注意力

### 3.6.1 堆叠多个单头注意力层
**多头注意力（Multi-Head Attention, MHA）** 允许模型同时关注输入序列的不同子空间。实现方法是并行计算多个独立的注意力头（head），然后将其输出拼接在一起：
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads=2):
        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
```

---

## 本章总结
- **注意力机制** 通过加权求和输入元素，生成包含全局信息的上下文向量。
- **自注意力** 计算每个输入元素相对于所有其他输入元素的重要性。
- **因果注意力** 通过掩码确保模型在生成文本时不会泄露未来信息。
- **多头注意力** 允许模型同时关注多个信息子空间，从而提升性能。





# 第四章 从零开始实现 GPT 模型以生成文本

Implementing a GPT model from scratch to generate text

在本章中，将基于前面章节所学的知识，逐步构建一个完整的 **GPT（生成式预训练变换器）** 模型。已经实现了 **多头注意力机制**，这是 LLM（大语言模型）的核心组件之一。现在，将实现其他构建模块，并将它们组装成一个类似 GPT 的模型，在下一章对其进行训练，使其能够生成类似人类的文本。

---

## 4.1 编写 LLM 体系结构

GPT（Generative Pre-trained Transformer）是一种大型深度神经网络架构，旨在逐个单词（或标记）生成新文本。尽管 GPT 规模庞大，但其模型架构并不复杂，因为其许多组件都是重复的，如 **Transformer 块**。

GPT 体系结构的主要组成部分：
1. **输入文本的标记化**（Tokenization）和 **嵌入**（Embedding）
2. **掩码多头自注意力**（Masked Multi-Head Attention）
3. **前馈神经网络层**（Feed Forward Network，FFN）
4. **层归一化**（Layer Normalization）
5. **残差连接**（Residual Connections）

我们将从一个 **占位 GPT 模型**（Dummy GPT Model）开始，实现一个简单的 GPT 结构，然后逐步填充其中的关键组件。

---

## 4.2 通过层归一化（Layer Normalization）稳定训练

在深度神经网络中，不同层的激活值可能会有较大波动，从而影响训练稳定性。**层归一化**（Layer Normalization, LN）是一种常见的技术，它通过对每个样本的隐藏状态进行归一化，来提高训练稳定性。

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

---

## 4.3 通过 GELU 激活函数实现前馈网络

在 Transformer 中，每个 **Transformer 块** 包含一个 **前馈神经网络**（FFN），用于对注意力层的输出进行进一步处理。GPT 采用 **GELU（高斯误差线性单元）** 作为激活函数，相比 ReLU，其可以更平滑地进行非线性转换。

```python
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
```

---

## 4.4 通过残差连接优化 Transformer 块

Transformer 采用 **残差连接**（Shortcut Connection），将输入直接加到输出上，以缓解梯度消失问题，提高深层网络的训练效果。

```python
class ResidualConnection(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(fn.net[0].in_features)

    def forward(self, x):
        return x + self.fn(self.norm(x))
```

---

## 4.5 连接注意力和前馈层，构建 Transformer 块

一个完整的 **Transformer 块** 由以下几个部分组成：
- **多头注意力层**（Multi-Head Attention）
- **前馈神经网络**（Feed Forward Network, FFN）
- **层归一化（LayerNorm）**
- **残差连接（Residual Connections）**

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, hidden_dim, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(dim, heads)
        self.residual_attention = ResidualConnection(self.attention)
        self.feed_forward = FeedForward(dim, hidden_dim, dropout)
        self.residual_ffn = ResidualConnection(self.feed_forward)

    def forward(self, x):
        x = self.residual_attention(x)
        x = self.residual_ffn(x)
        return x
```

---

## 4.6 编写 GPT 模型

现在可以组合所有的组件，实现完整的 **GPT 架构**：
- **嵌入层**（Embedding Layer）：将输入标记转换为向量表示。
- **多个 Transformer 块**（Transformer Blocks）：核心计算部分。
- **最终层归一化**（Final LayerNorm）：稳定输出。
- **输出层**（Output Layer）：预测下一个标记。

```python
class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_emb = nn.Dropout(config["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config["emb_dim"], config["n_heads"],
                               config["hidden_dim"], config["drop_rate"])
              for _ in range(config["n_layers"])]
        )
        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = self.drop_emb(tok_embeds + pos_embeds)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

---

## 4.7 生成文本

GPT 通过 **自回归（Autoregressive）** 方式逐步生成文本，每次预测下一个标记，并将其追加到输入中。

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

---

## 本章总结

- 我们从头实现了一个 **GPT 架构**，包括 **嵌入层、Transformer 块、层归一化、前馈网络、残差连接**。
- 通过 **多层 Transformer 块** 构建深度模型，使其能够生成文本。
- 采用 **GELU 激活、LayerNorm、残差连接** 提高训练稳定性。
- 设计了一个 **文本生成循环**，使 GPT 能够 **逐个标记生成文本**。

在下一章，我们将训练这个 GPT 模型，使其能够生成高质量的自然语言文本！





# 第五章 在无标签数据上进行预训练

Pretraining on Unlabeled Data

在前几章中，我们构建了 GPT 模型的核心架构。现在，将进入 **预训练（Pretraining）** 阶段，使模型学习语言结构，以便能够生成连贯的文本。在预训练过程中，将使用 **大规模无标签文本数据**，并采用 **自监督学习（self-supervised learning）** 方法让模型自己生成标签进行训练。

---

## 5.1 预训练的目标

LLM（大语言模型）预训练的核心目标是：
1. **学习语言结构**：让模型理解单词、短语、句子及其关系。
2. **掌握上下文信息**：使模型能够在较长的文本序列中捕捉语义信息。
3. **为后续任务提供基础**：在通用数据上预训练后，模型可以被微调（fine-tuning）以执行特定任务，如翻译、分类或问答。

我们主要训练模型进行 **下一个词预测（Next Word Prediction）**，即：
$$
P(w_t | w_1, w_2, ..., w_{t-1})
$$
其中，$ w_t $ 是当前时间步的单词，模型根据之前的单词预测它。

---

## 5.2 训练 LLM 的数据集

训练 LLM 需要庞大的数据集。例如：
- **CommonCrawl**（4100 亿个标记，约 570GB）
- **Wikipedia**（3 亿个标记，约 3GB）
- **BooksCorpus**（超过 1 亿个标记）
- **ArXiv 论文**（92GB，技术性文本）
- **StackExchange 代码相关问答**（78GB）。

我们将使用 **小型数据集** 进行实验，以便能够在 **个人计算机** 上运行，而无需超算集群。

---

## 5.3 训练循环的实现

### 5.3.1 训练流程

GPT 预训练采用 **标准深度学习训练循环**：
1. 遍历数据集 **(epochs)**
2. 逐批次处理数据 **(batches)**
3. 计算损失 **(loss function)**
4. 反向传播 **(backpropagation)**
5. 更新模型参数 **(optimizer step)**
6. 监控训练损失和验证损失 **(monitor training progress)**。

### 5.3.2 训练代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=3):
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            logits = model(input_batch.to(device))
            loss = loss_fn(logits.view(-1, logits.shape[-1]), target_batch.view(-1).to(device))
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
```

---

## 5.4 评估生成文本的质量

在预训练过程中，我们需要评估模型生成的文本质量。常见方法包括：
- **训练损失（Training Loss）**：衡量模型在训练数据上的表现。
- **验证损失（Validation Loss）**：测试模型在未见过的数据上的泛化能力。
- **困惑度（Perplexity, PPL）**：
  $$
  PPL = e^{\text{Loss}}
  $$
  较低的 PPL 值表示更好的语言建模能力。

---

## 5.5 预训练成本分析

训练一个 **完整的 LLM** 需要 **大量计算资源**。例如：
- **LLaMA 2 (7B 参数) 训练耗时**：184,320 GPU 小时（8× A100 GPU）
- **训练成本**：约 **$690,000 美元**（基于 AWS 价格）。

可以选择：
1. **使用小规模模型进行实验**
2. **加载公开的预训练权重（如 OpenAI 提供的 GPT-2 权重）** 以节省训练成本。

---

## 5.6 加载预训练模型权重

可以直接加载 GPT-2 的预训练权重，而无需从头训练整个模型：

```python
import torch
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
```

这样，我们可以在 **本地计算机** 运行 GPT-2，而不必消耗数千小时的 GPU 计算资源。

---

## 5.7 预训练的下一步

- **模型微调（Fine-Tuning）**：
  - 在特定领域数据上训练，如法律、医学或编程。
- **任务特定优化**：
  - 例如，优化 **摘要（Summarization）** 或 **问答（Question Answering）**。
- **部署 LLM**：
  - 使用 **ONNX、CoreML** 等工具优化模型，使其能在移动设备上运行。

---

## 本章总结

- 预训练 LLM **需要大规模数据**，但我们可以用小型数据集进行实验。
- 训练循环包括 **计算损失、梯度更新和评估生成质量**。
- **计算成本昂贵**，但我们可以 **加载 GPT-2 预训练权重** 来节省时间和资源。
- 预训练的模型可以 **进一步微调**，以适应特定任务。

在下一章，我们将讨论如何 **微调预训练的 LLM** 以执行特定任务，如文本分类和指令跟随（Instruction Following）！





# 第六章 用于分类的微调

Fine-Tuning for Classification

在前几章中，我们已经完成了 **LLM（大语言模型）** 的基础架构搭建、预训练，并加载了预训练权重。现在，我们将对 LLM 进行 **微调（Fine-Tuning）**，使其能够执行文本分类任务。我们将以 **垃圾短信（spam）分类** 为示例，展示如何将 LLM 调整为专用分类器。

---

## 6.1 不同类别的微调

LLM 的微调方法主要分为：
1. **指令微调（Instruction Fine-Tuning）**：训练模型根据自然语言指令执行任务，如问答、翻译等。
2. **分类微调（Classification Fine-Tuning）**：训练模型识别特定类别，例如垃圾邮件分类、情感分析等。

在本章中，我们重点关注 **分类微调**，即让模型能够在 **垃圾短信（spam）** 和 **非垃圾短信（not spam）** 之间进行分类。

---

## 6.2 数据集准备

我们将使用 **垃圾短信分类数据集**，该数据集包含：
- **垃圾短信（spam）**
- **非垃圾短信（not spam）**

首先，我们需要 **下载和预处理数据**，然后转换为 **PyTorch 数据加载器（DataLoader）** 以便进行训练。

```python
import torch
from torch.utils.data import Dataset, DataLoader

class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokenized = self.tokenizer.encode(self.texts[idx], max_length=self.max_length, truncation=True)
        return torch.tensor(tokenized), torch.tensor(self.labels[idx])

# 示例数据
texts = ["Congratulations! You've won a prize.", "Hey, are we still on for lunch?"]
labels = [1, 0]  # 1 表示垃圾短信，0 表示非垃圾短信

# 加载数据
dataset = SpamDataset(texts, labels, tokenizer, max_length=128)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
```

---

## 6.3 修改 LLM 以支持分类任务

在预训练的 LLM 中，输出层用于预测 **下一个 token**，但在分类任务中，我们需要预测 **类别标签**。因此，我们需要 **替换输出层**，使其适应分类任务。

```python
import torch.nn as nn

# 冻结 LLM 的所有参数
for param in model.parameters():
    param.requires_grad = False

# 替换输出层
num_classes = 2  # 二分类任务
model.out_head = nn.Linear(in_features=768, out_features=num_classes)

# 只训练新的输出层
for param in model.out_head.parameters():
    param.requires_grad = True
```

这样，我们就 **只训练输出层**，而不会修改 LLM 的其余部分。这种方法被称为 **部分微调（Partial Fine-Tuning）**，它比全模型微调更高效。

---

## 6.4 计算分类损失

在分类任务中，我们通常使用 **交叉熵损失（Cross-Entropy Loss）** 作为目标函数。

```python
loss_fn = nn.CrossEntropyLoss()

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # 只取最后一个 token 的输出
    loss = loss_fn(logits, target_batch)
    return loss
```

---

## 6.5 计算分类准确率

CoreML任务的核心是 **评估准确率（Accuracy）**，我们可以通过 **argmax** 计算预测的类别，并统计正确率。

```python
def calc_accuracy_loader(data_loader, model, device):
    model.eval()
    correct, total = 0, 0

    for input_batch, target_batch in data_loader:
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)

        with torch.no_grad():
            logits = model(input_batch)[:, -1, :]  # 取最后一个 token
            predicted_labels = torch.argmax(logits, dim=-1)

        correct += (predicted_labels == target_batch).sum().item()
        total += target_batch.size(0)

    return correct / total
```

---

## 6.6 训练 LLM 进行分类

我们将训练 LLM，使其能够进行垃圾短信分类。训练循环包括：
1. **计算损失**
2. **反向传播**
3. **更新模型参数**
4. **评估训练损失和准确率**。

```python
def train_classifier(model, train_loader, val_loader, optimizer, device, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
        
        train_acc = calc_accuracy_loader(train_loader, model, device)
        val_acc = calc_accuracy_loader(val_loader, model, device)
        print(f"Epoch {epoch+1}: Train Acc = {train_acc:.2f}, Val Acc = {val_acc:.2f}")
```

---

## 6.7 评估微调后的 LLM

训练完成后，我们可以使用 **测试数据** 评估 LLM。

```python
text = "You are a winner! Claim your prize now."
input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

with torch.no_grad():
    logits = model(input_ids)[:, -1, :]
    predicted_label = torch.argmax(logits).item()

print("Predicted label:", "Spam" if predicted_label == 1 else "Not Spam")
```

---

## 6.8 保存和加载微调后的模型

为了复用训练好的分类器，我们可以将其保存，并在需要时重新加载。

```python
torch.save(model.state_dict(), "spam_classifier.pth")

# 加载模型
model.load_state_dict(torch.load("spam_classifier.pth"))
model.eval()
```

---

## 本章总结

- **LLM 微调主要有两种方式**：
  1. **指令微调（Instruction Fine-Tuning）**
  2. **分类微调（Classification Fine-Tuning）**
- **数据准备**：转换数据集为 PyTorch DataLoader
- **修改 LLM 以支持分类**：替换输出层
- **计算分类损失与准确率**：使用 **交叉熵损失** 和 **argmax 分类**
- **训练 LLM 进行分类**：采用标准训练循环
- **评估微调后的模型**：计算分类准确率
- **保存和加载微调后的 LLM**。

在下一章，我们将探讨 **指令微调（Instruction Fine-Tuning）**，使 LLM 能够执行更加复杂的任务，如对话系统和智能助手！





# 第七章 微调 LLM 以遵循指令

Fine-Tuning to Follow Instructions

在之前的章节中，我们实现了 **GPT 架构**，并通过 **分类微调** 使其能够执行垃圾短信检测任务。在本章中，我们将深入探讨 **指令微调（Instruction Fine-Tuning）**，这是一种用于训练 LLM（大语言模型）理解并执行自然语言指令的方法。指令微调广泛用于 **对话系统、智能助手和 AI 任务自动化**。

---

## 7.1 指令微调简介

虽然预训练的 LLM 具备文本生成能力，但它们通常难以正确理解并执行复杂指令。例如：
- **输入**："请将以下句子转换为被动语态：'The chef cooks the meal every day.'"
- **LLM 输出（错误示例）**："The chef cooks the meal every day. Convert the active sentence to passive."
- **理想输出**："The meal is cooked by the chef every day."

这说明预训练模型并未正确执行指令，而是部分复述了输入文本。因此，我们需要 **指令微调** 来增强模型的指令理解能力。

---

## 7.2 监督式指令微调数据集准备

指令微调需要一个 **指令-响应对（instruction-response pairs）** 作为训练数据。例如：
- **指令**："将 45 公里转换为米。"
- **响应**："45 公里等于 45000 米。"

本书的实验数据集包含 **1100 对指令-响应样本**，但你也可以使用 **开源指令数据集**（如 OpenChat、UltraChat）。

我们将数据划分为：
- **训练集（85%）**
- **验证集（10%）**
- **测试集（5%）**

```python
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]
```

---

## 7.3 构造训练批次

为了提高训练效率，我们需要 **数据填充（padding）** 使所有训练样本长度一致，并使用 **PyTorch DataLoader** 进行批量训练。

```python
import torch
from torch.utils.data import DataLoader, Dataset

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.encoded_texts = [tokenizer.encode(entry["instruction"] + entry["response"]) for entry in data]

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, idx):
        return self.encoded_texts[idx]

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
```

---

## 7.4 加载预训练 LLM 进行微调

我们将使用 **GPT-2 Medium（355M 参数）** 作为基础模型，并用 **AdamW 优化器** 进行微调。

```python
from transformers import GPT2LMHeadModel, AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
```

---

## 7.5 计算损失函数

在指令微调中，我们采用 **交叉熵损失（Cross-Entropy Loss）**，并忽略填充部分（padding tokens）。

```python
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

def calc_loss(input_batch, target_batch, model):
    logits = model(input_batch).logits[:, :-1, :]
    loss = loss_fn(logits.view(-1, logits.shape[-1]), target_batch.view(-1))
    return loss
```

---

## 7.6 训练 LLM 以遵循指令

### 7.6.1 训练循环
我们使用前面定义的 **数据加载器、损失函数和优化器** 进行 LLM 训练。

```python
def train_instruction_model(model, train_loader, val_loader, optimizer, device, epochs=2):
    for epoch in range(epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss(input_batch.to(device), target_batch.to(device), model)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

train_instruction_model(model, train_loader, val_loader, optimizer, device)
```

### 7.6.2 训练时间分析
在不同设备上的训练时间：
| 模型         | 设备                 | 训练时间（2 轮） |
| ------------ | -------------------- | ---------------- |
| GPT-2 Medium | CPU (M3 MacBook Air) | 15.78 min        |
| GPT-2 Medium | GPU (NVIDIA L4)      | 1.83 min         |
| GPT-2 Small  | GPU (NVIDIA A100)    | 0.39 min         |

建议使用 **GPU 训练**，以加速训练过程。

---

## 7.7 评估微调后的 LLM

### 7.7.1 生成指令响应
我们让 LLM 生成 **符合指令的输出**，并进行评估。

```python
text = "请将以下句子转换为被动语态：'The chef cooks the meal every day.'"
input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(input_ids, max_length=50)
print("Generated response:", tokenizer.decode(output[0]))
```

**示例输出**：
```
"The meal is cooked by the chef every day."
```

### 7.7.2 量化评估
我们计算 **BLEU 分数** 或 **ROUGE 分数** 评估生成文本的质量：
```python
from sacrebleu import corpus_bleu

def evaluate_bleu(reference, hypothesis):
    return corpus_bleu(hypothesis, [reference]).score

reference = ["The meal is cooked by the chef every day."]
hypothesis = ["The meal is prepared by the chef every day."]
print("BLEU Score:", evaluate_bleu(reference, hypothesis))
```

---

## 7.8 结论与后续优化

### 7.8.1 训练后优化
- **超参数调整**（调整学习率、批次大小）
- **扩大数据集**（使用更多指令数据）
- **实验不同的提示格式**（如 Alpaca、Phi-3 格式）。

### 7.8.2 未来发展
- **偏好微调（Preference Fine-Tuning）**：让 LLM 生成更符合人类偏好的回复。
- **强化学习（RLHF）**：结合人类反馈进一步优化指令执行质量。

---

## 本章总结

- **指令微调** 使 LLM 能够 **理解并执行自然语言指令**。
- **数据集准备**：使用 **指令-响应对** 训练 LLM。
- **训练流程**：使用 **GPT-2 Medium** 并 **优化损失函数** 进行训练。
- **评估方法**：基于 **BLEU、ROUGE** 量化 LLM 生成质量。
- **进一步优化**：通过 **超参数调整、RLHF** 提高 LLM 的指令理解能力。
- 用Ollama部署 LLM。





 