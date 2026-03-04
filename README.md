# KQA-Pro Hallucination Lab

## 项目定位

这是一个围绕 **KQA-Pro** 的独立整理仓库，目标是回答一个核心问题：

> 基于知识图谱推理过程构造出的 `entity`、`head`、`reason` 三项指标，是否能够作为 **hallucination detection / hallucination signal** 的有效依据，并反映大模型在 KQA-Pro 上的真实答题能力？

更具体地说，我们希望证明这些图推理指标与下游答题准确率 **正相关**，从而说明它们能够作为检测或刻画幻觉风险的代理信号。

---

## 研究背景

这个项目分为两部分：

### 1. 仓库外部的答题评测

在本仓库之外，使用 `lm-evaluation-harness` 对 KQA-Pro 验证集进行了评测，评测对象包括：

- `Llama-3.1-8B`
- `Llama-3.1-8B-Instruct`

评测任务包括：

- `MCQ`
- `QA`

由此得到了**每个问题级别**的答题结果指标，例如：

- `acc`
- `exact_match`
- `contains`

这些指标作为下游任务表现的观测值。

### 2. 本仓库内部的图推理指标构建

在本仓库中，我们对 KQA-Pro 数据集做了大量预处理和重构，核心工作包括：

- 解析 KQA-Pro 的 `SPARQL`
- 构建 gold 子图
- 构建 1-hop 候选边
- 设计问题理解任务与问题推理任务
- 使用 `vLLM` 对 `Llama-3.1-8B-Instruct` 运行图相关评测

最终得到三项图推理指标：

- `entity`
- `head`
- `reason`

---

## 三项指标的含义

### `entity`

衡量模型是否能从问题中识别出正确的关键实体。

### `head`

衡量模型是否能从候选实体中选出正确的推理起点。

### `reason`

衡量模型是否能在局部候选边中选出真正支持问题求解的推理边。

---

## 核心假设

我们的目标不是单纯构造新指标，而是验证以下假设：

> 如果一个问题的 `entity / head / reason` 指标更高，那么模型在该问题上的答题准确率也应该更高；反过来，这些指标较低时，更容易出现 hallucination 或错误回答。

也就是说，我们期待看到：

- `entity` 与 `acc / exact_match / contains` 正相关
- `head` 与 `acc / exact_match / contains` 正相关
- `reason` 与 `acc / exact_match / contains` 正相关

并进一步希望复合指标也能反映答题能力，并作为 hallucination signal 使用。

---

## 当前进展

目前已经完成或部分完成：

- KQA-Pro 的 SPARQL 解析与 gold 子图构建
- qualifier / 1-hop 子图相关处理
- 问题理解评测脚本
- 问题推理评测脚本
- `entity / head / reason` 指标聚合
- 与下游答题结果的相关性分析
- 按 `typ`、按结构聚类的相关性探索

---

## 当前问题

目前还**没有稳定地证明**三项指标与答题准确率之间存在理想的正相关关系，因此“它们能否作为有效 hallucination signal”这个核心问题仍未解决。

现阶段的主要困难包括：

- `entity` 与准确率的相关性偏弱
- `head` 与准确率的相关性不稳定
- `reason` 的聚合方式可能与最终答题表现不完全匹配
- 不同模型、不同提示词、不同 shot 设置会显著影响结果
- 不同题型混合后，整体相关性可能被稀释
- 现有老仓库积累了太多实验性脚本，结构已不适合继续迭代

因此需要新建一个更清晰的仓库，重新整理研究流程。

---

## 新仓库的目标

这个新仓库将用于系统整理以下内容：

1. KQA-Pro 数据处理流程
2. gold 子图与 1-hop 子图构建逻辑
3. 图推理评测脚本
4. 指标聚合脚本
5. 样本级 / 类型级 / 聚类级相关性分析
6. 可视化与结果解释
7. 实验配置与结果溯源
8. 幻觉检测假设的验证与失败案例分析

---

## 当前建议的工作重点

后续优先处理的问题：

1. 统一答题侧与图评测侧的模型、shot、数据版本
2. 明确每份结果文件的 provenance
3. 重新审视 `reason` 的聚合方式
4. 继续分析 `entity / head / reason` 与下游指标的关系
5. 区分不同题型、结构复杂度、聚类结果下的相关性

---

## 数据流水线重构

当前仓库已经开始用纯 Python 重构旧的 `kqa_v5 / kqa_v6` notebook 流程，新的数据产物改为三张语义化表：

- `question_base.csv`
- `gold_subgraphs.csv`
- `onehop_by_seed.csv`

目录结构如下：

```text
kqapro-hallucination-lab/
  scripts/
    export_question_base.py
    build_gold_subgraphs.py
    build_onehop_by_seed.py
    build_all_processed.py
  src/kqapro_hallucination/
    io.py
    schemas.py
    kb_loader.py
    sparql_engine.py
    gold_subgraph_builder.py
    onehop_builder.py
    literal_utils.py
    paths.py
  data/processed/
```

### 三张表的职责

- `question_base.csv`
  - 基础问题表
  - 包含 `idx/question/typ/choices/answer/q_ent/.../sparql`
- `gold_subgraphs.csv`
  - 由 SPARQL 实际执行得到的 gold 子图表
  - 包含 `gold_subgraph_edges / gold_heads / gold_tails / gold_entities`
- `onehop_by_seed.csv`
  - 基于 gold 子图中出度不为 0 的节点构造的 1-hop 候选表
  - 包含 `onehop_by_seed`

### 运行方式

单步运行：

```bash
python scripts/export_question_base.py
python scripts/build_gold_subgraphs.py
python scripts/build_onehop_by_seed.py
```

一键运行：

```bash
python scripts/build_all_processed.py
```

如果你希望显式指定输入输出路径，可以这样运行：

```bash
python scripts/export_question_base.py \
  --input_path ../data/kqa_v2.csv \
  --output_path data/processed/question_base.csv

python scripts/build_gold_subgraphs.py \
  --base_path data/processed/question_base.csv \
  --output_path data/processed/gold_subgraphs.csv \
  --checkpoint_path data/processed/gold_subgraphs.checkpoint.json

python scripts/build_onehop_by_seed.py \
  --gold_path data/processed/gold_subgraphs.csv \
  --output_path data/processed/onehop_by_seed.csv
```

说明：

- 脚本默认优先读取本仓库 `data/` 下的原始依赖。
- 如果本仓库 `data/` 下没有原始文件，会回退尝试读取旧工作区中的 `../data/` 与 `../../kqa-pro/dataset/kb.json`。
- 产物统一写入本仓库自己的 `data/processed/`。
- `data/processed/*.csv` 属于派生产物，默认不纳入 git 跟踪。

---

## 备注

- 本仓库是从旧仓库中拆分出来的“干净版本”起点。
- 当前阶段主要目标是**整理问题定义、实验目标、指标体系和分析路线**。
- 代码和数据会逐步迁移，不追求一次性搬完。
