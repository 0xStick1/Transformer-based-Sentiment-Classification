# Transformer-based Chinese Sentiment Classification (From Scratch) 🚀

这是一个深度探索 Transformer 架构在中文语义理解中应用的研究项目（判断句子情绪积极 or 消极）。本项目坚持**全流程自研**（从分词算法、模型架构到训练策略），在不依赖任何外部预训练大模型（如 BERT Base）的情况下，通过极致的策略优化，使仅有 4 层的主干架构在近 20 万规模的真实中文评论数据上达到 **94.62%** 的工业级准确率。

## 🧪 项目核心愿景
*   **全链路透明**：手动实现 Transformer Encoder、Multi-Head Attention 及 Pre-LayerNorm 架构。
*   **端到端自主化**：自研 BPETokenizer 分词引擎，支持动态掩码与全词掩码 (WWM) 训练。
*   **性能压榨**：研究在大规模、多领域（购物+外卖+社交）数据集下的模型泛化能力与收敛极限。

---

## 🏗️ 架构演进日志 (Architecture Evolution)

### Phase 1-3: 基础与结构升级
*   **BPE 整合**：引入 5000 词表 BPE 分词，平衡语义表达与参数效率。
*   **BERT-style**：引入 `[CLS]` 标记作为全句总结符，并采用 **Pre-LayerNorm** 架构确保深层网络训练的稳定性。
*   **SWA**：通过随机权重平均 (Stochastic Weight Averaging) 平滑模型轨迹，消除指标震荡。

### Phase 4: MLM 技术 (Advanced Masking)
*   **Dynamic Masking**：每一轮训练随机更换掩码位置，数据利用率提升 100%。
*   **Whole Word Masking (WWM)**：自研启发式中文分词算法，强制抠掉整个词组（如“物流快”），迫使模型理解深层语言逻辑。

### Phase 5: 数据添加 (Add Data)
*   **规模增加**：整合购物、外卖、微博三大语境，数据量由 6 万升至 **19.3 万**。
*   **预训练增加**：预训练由原来的 50 轮增加至 100 轮，充分吸收多领域语义。

---

## 📂 核心文件

*   `model_scratch.py`: Transformer 核心架构实现。
*   `tokenizer_bpe.py`: 自研 BPE 分词引擎。
*   `merge_data.py`: 数据自动化采集与清洗合体工具。
*   `pretrain.py`: WWM + 动态掩码预训练引擎。
*   `finetune.py`: SWA + 差分学习率微调工具。
*   `predict_finetuned.py`: 推理界面（含置信度过滤）。

---

## 🚀 快速上手 (Quick Start)

### 1. 环境准备
```bash
pip install torch pandas numpy tqdm requests
```

### 2. 下载数据 (19.3 万样本)
```bash
python merge_data.py
```

### 3. 三步走训练流程
1.  **预训练**：`python pretrain.py`
2.  **情感微调**：`python finetune.py` (适配二分类任务，建议 60 Epochs)
3.  **人机交互测试**：`python predict_finetuned.py` (非自己训练请从 Release 下载所需文件)

---

## 📊 性能里程碑 (Benchmarks)

| 训练阶段     | 数据规模 | 核心技术 | 准确率 (Val) |
|:---------| :--- | :--- | :--- |
| **基础阶段** | 6.2万 | Mean Pooling | 87.41% |
| **优化阶段** | 6.2万 | [CLS] + Pre-Norm + SWA | 91.25% |
| **当前阶段** | **19.3万** | **WWM + Marathon Pretrain** | **94.62%** |

---

## 📽️ 模型部署 (Deployment)

本项目支持将训练好的 PyTorch 模型导出为 **TorchScript** 格式。

### 导出模型
```bash
python export_model.py
```
这会生成 `model_torchscript.pt`，可在无 Python 环境或 C++ 环境中加载。

---
## 📄 License
MIT License. 欢迎在该项目基础上进行二次实验与学术研究。
