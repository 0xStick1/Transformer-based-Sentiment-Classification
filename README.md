# Transformer-based Chinese Sentiment Classification (From Scratch) 🚀

这是一个深入探索 Transformer 架构在中文语义理解中应用研究项目。本项目旨在通过**全流程自研**（从分词算法到底层架构优化），研究模型在处理真实世界复杂评论数据时的内在逻辑。

## 🧪 项目核心愿景
*   **透视底层细节**：手动实现 Transformer Encoder。
*   **端到端自主化**：从 Byte Pair Encoding (BPE) 分词器的训练，到基于 MLM 的大规模语料预训练，再到特定领域的下游微调。
*   **极限性能挑战**：在非预训练大模型背景下，通过策略优化，使仅有 4 层的主干架构在复杂中文数据集上达到 **91.25%** 的工业级准确率。

---

## 🏗️ 架构演进日志 (Architecture Evolution)

### Phase 1: 基础构建 (The Baseline)
使用简单的字符级分词与 Global Mean Pooling。初步验证了 Transformer 对中文社交语言的适应性，但受限于分词粒度，模型难以捕捉复合词语义。

### Phase 2: 分词进化 (BPE Integration)
引入自研 **BPETokenizer**。
*   **权衡分析**：在 5000 词表与 8000 词表之间进行了多次实验。结论显示，针对 6 万条数据，5000 词表的 Embedding 密度更高，过拟合风险更低，能够更好地平衡泛化能力与语义表达。

### Phase 3: 架构升级 (BERT-style Implementation)
为了冲击 91%+ 的性能瓶颈，进行了三大硬核改动：
1.  **[CLS] 标记引入**：放弃 Mean Pooling，采用 learnable `[CLS]` 标记作为全句情感总结符，显著提升了模型对重难点情感词的敏感度。
2.  **Pre-LayerNorm 结构**：遵循现代深层模型（如 GPT-3）的最佳实践。将 Norm 提前至 Attention/FFN 之前。实验证明，这让模型在深度增加到 4 层以上时，训练收敛曲线更加平滑。
3.  **SWA (Stochastic Weight Averaging)**：在微调末期引入权重平均。通过平均最后 10 个 Epoch 的模型轨迹，成功抹平了验证集的震荡，稳定提升了约 1.5% 的准确率。

---

## 📊 技术细节深挖 (Technical Deep-Dive)

### 1. 差分学习率策略 (Discriminative Fine-tuning)
虽然 Encoder 是自研预训练的，但为了保护模型已具备的通用语义，对不同层应用了“温差学习”：
*   **Encoder LR**: `1e-5` (保护特征提取能力)
*   **Classification Head LR**: `1e-4` (加速下游任务适配)

### 2. 多领域数据集挑战
本项目使用全量 `online_shopping_10_cats` 评论数据。模型需要同时处理书籍、手机、生活用品等 10 个领域的语言风格，对模型的跨领域特征提取能力提出了极高要求。

---

## 📂 核心文件矩阵

*   `model_scratch.py`: 核心架构定义。实现了 **Pre-Norm** Transformer 块与 **[CLS]** 逻辑。
*   `tokenizer_bpe.py`: 硬核分词引擎。实现了一种能够处理中文字符偏移的子词合并算法。
*   `pretrain.py`: MLM 预训练流水线。包括动态权重衰减与 Cosine Warmup 策略。
*   `finetune.py`: 高级微调引擎。集成了 **SWA** 与差分学习率优化逻辑。

---

## 🚀 快速上手 (Quick Start)

### 环境配置
```bash
pip install torch pandas numpy tqdm
```

### 生产流水线
1.  **预训练基石**：`python pretrain.py` (获取具备语义常识的 Encoder)
2.  **情感对齐**：`python finetune.py` (获取 SWA 加持的分类模型)
3.  **人机交互测试**：`python predict_finetuned.py`

---
## 📄 License
MIT License. 欢迎在该项目基础上进行二次实验与学术研究。
