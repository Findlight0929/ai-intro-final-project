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

## 推荐目录结构

```text
ai-intro-final-project/
├─ rumer2026/                # 原始数据集
├─ src/                      # 主要代码
├─ models/                   # 训练得到的模型文件
├─ outputs/                  # 预测结果、实验输出
├─ notebooks/                # 数据分析与实验草稿
├─ report/                   # 报告和插图
├─ README.md
├─ requirements.txt
├─ main.py
└─ 小组分工方案.md
```

## 当前代码说明

当前已提供最小可开工框架：

- `src/config.py`：路径与常量配置
- `src/preprocess.py`：基础文本清洗
- `src/train_baseline.py`：baseline 训练脚本
- `src/predict.py`：单条/批量预测脚本
- `src/explain.py`：解释文本生成模块
- `main.py`：统一入口，用于单条文本预测与解释

## 环境安装

建议使用 Python 3.10 及以上版本。

安装依赖：

```bash
pip install -r requirements.txt
```

## 训练 baseline

```bash
python src/train_baseline.py
```

运行后将：

- 读取 `rumer2026/train.csv` 和 `rumer2026/val.csv`
- 训练 `TF-IDF + Logistic Regression` baseline
- 在验证集上输出 Accuracy 和分类报告
- 将模型保存到 `models/`
- 将验证集预测结果保存到 `outputs/`

## 单条文本预测

```bash
python main.py --text "Breaking news example text"
```

输出格式示例：

```text
Label: 1
Reason: The text is classified as rumor because it uses urgent or emotionally charged wording and does not provide a verifiable source.
```

## 小组协作建议

- 组长负责仓库维护、系统整合、README 和报告统稿
- 建模同学负责 baseline 和模型改进
- 解释模块同学负责判断依据生成和案例整理
- 每位成员都应直接提交代码或文档，保留清晰 commit 记录

## 当前待推进事项

1. 跑通 baseline 并记录验证集结果
2. 优化英文文本预处理与模型效果
3. 补充更高质量的解释模块
4. 完善报告和案例展示
