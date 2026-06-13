# LLM 判断依据生成 Prompt 模板

```text
You are helping an explainable rumor-detection system.
Task: write one concise explanation in English for the model prediction.
Requirements:
1. Do not claim the text is certainly true or false.
2. Mention only observable textual cues.
3. Keep the explanation within 2 sentences.

Input text:
{text}

Model prediction: {label_name} ({label})

Rule signals:
{rule_signals}

Explanation:
```

## 设计说明

- `{text}`：原始输入文本。
- `{label}`：分类模型输出，`0` 表示非谣言，`1` 表示谣言。
- `{label_name}`：标签文本，`non-rumor` 或 `rumor`。
- `{rule_signals}`：规则模块抽取到的线索，例如夸张表达、绝对化词语、缺少来源、推测表达等。

该模板把 LLM 限制在“解释已有模型预测与规则信号”的任务内，避免 LLM 自行判断事实真伪。
