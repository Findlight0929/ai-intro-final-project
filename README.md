> This repository is a fork of a team course project for Introduction to
> Artificial Intelligence at Shanghai Jiao Tong University.
>
> My primary contribution was the explainability layer, including rule-based
> explanations, TF-IDF retrieval, error analysis, and integration with the
> classification workflow.
# ai-intro-final-project

人工智能导论大作业项目仓库，题目为**可解释的谣言检测**。

## 项目目标

构建一个能够对输入文本进行谣言分类并输出判断依据的系统。

系统输入一段英文社交媒体文本，输出：

- `label`：分类结果，`0` 表示非谣言，`1` 表示谣言
- `reason`：一段自然语言说明判断依据

## 当前数据

仓库当前已经包含课程数据集：

```text
rumer2026/
├─ train.csv
└─ val.csv
```

已确认字段包括：

- `id`
- `text`
- `label`
- `event`

## 项目结构

```text
ai-intro-final-project/
├─ rumer2026/                # 原始数据集
├─ src/                      # 主要代码
│  ├─ config.py              # 路径与字段配置
│  ├─ preprocess.py          # 文本清洗
│  ├─ train_baseline.py      # baseline 训练与验证
│  ├─ predict.py             # 单条/批量预测
│  ├─ rag.py                 # RAG 相似训练样例检索
│  └─ explain.py             # 判断依据生成
├─ models/                   # 训练得到的模型文件
├─ outputs/                  # 预测结果、实验指标、解释样例
├─ notebooks/                # 数据分析与实验草稿
├─ report/                   # 报告和插图
├─ README.md
├─ requirements.txt
├─ main.py                   # 单条文本预测入口
└─ 小组分工方案.md
```

## 环境安装

建议使用 Python 3.10 及以上版本。

安装依赖：

```bash
pip install -r requirements.txt
```

## 训练 baseline

```bash
python -m src.train_baseline
```

运行后将：

- 读取 `rumer2026/train.csv` 和 `rumer2026/val.csv`
- 训练 `TF-IDF + Logistic Regression` baseline
- 在验证集上输出 Accuracy 和分类报告
- 将模型保存到 `models/baseline_pipeline.joblib`
- 将验证集预测结果保存到 `outputs/val_predictions.csv`
- 将前 20 条解释样例保存到 `outputs/examples.csv`
- 将指标保存到 `outputs/metrics.txt`

## 单条文本预测

训练完成后运行：

```bash
python main.py --text "Breaking news example text"
```

输出格式示例：

```text
Label: 1
Reason: The text is classified as rumor because it contains exaggerated, urgent, or emotionally provocative wording, such as breaking; and it does not provide an explicit source or confirmation cue. RAG context: 2/3 nearest training examples share the predicted rumor label, while the rest show mixed context.
```

## 批量预测

训练完成后，也可以对 CSV 文件批量预测。输入文件需要包含 `text` 字段。

```bash
python -m src.predict --input rumer2026/val.csv --output outputs/batch_predictions.csv
```

## 当前最小可运行版本说明

当前版本已经初步完成：

- 成员 2 任务：数据读取、文本预处理、baseline 训练、验证集评估、预测结果保存
- 成员 3 任务：基于规则 + RAG 的解释模块、单条解释输出、验证集解释样例保存

后续可以继续优化：

1. 对比 SVM、Naive Bayes、BERT 等模型
2. 调整文本预处理和 TF-IDF 参数
3. 使用模型高权重词增强解释质量
4. 接入学校提供的大语言模型接口生成更自然的解释
5. 在报告中加入错误案例分析和典型案例展示

## 小组协作建议

- 组长负责仓库维护、系统整合、README 和报告统稿
- 建模同学负责 baseline 和模型改进
- 解释模块同学负责判断依据生成和案例整理
- 每位成员都应直接提交代码或文档，保留清晰 commit 记录
