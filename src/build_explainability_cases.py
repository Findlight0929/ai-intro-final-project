from __future__ import annotations

import pandas as pd

from src.config import (
    EVENT_COLUMN,
    ID_COLUMN,
    LABEL_COLUMN,
    OUTPUTS_DIR,
    TEXT_COLUMN,
    VAL_PREDICTIONS_FILE,
)
from src.explain import analyze_text_signals
from src.predict import predict_dataframe


EXPLAINABILITY_CASES_FILE = OUTPUTS_DIR / "explainability_cases.csv"


def _active_signal_names(text: str) -> str:
    signals = analyze_text_signals(text)
    names = [signal.name for signal in signals if signal.has_hits]
    return "; ".join(names)


def _short_analysis(row: pd.Series) -> str:
    true_label = int(row[LABEL_COLUMN])
    prediction = int(row["prediction"])

    if true_label == prediction == 1:
        return "Correct rumor case: the explanation highlights suspicious wording and missing source cues."
    if true_label == prediction == 0:
        return "Correct non-rumor case: the explanation highlights weaker rumor cues or clearer reporting cues."
    if true_label == 0 and prediction == 1:
        return "False positive case: social-media style or emotional wording may have made a non-rumor look rumor-like."
    return "False negative case: the text may lack obvious rumor cues, so the model underestimates its risk."


def _select_cases(dataframe: pd.DataFrame, case_count: int = 10) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe["is_correct"] = dataframe[LABEL_COLUMN].astype(int) == dataframe["prediction"].astype(int)
    dataframe["case_group"] = dataframe.apply(
        lambda row: f"true_{int(row[LABEL_COLUMN])}_pred_{int(row['prediction'])}",
        axis=1,
    )

    selected_parts = []
    for group_name in [
        "true_1_pred_1",
        "true_0_pred_0",
        "true_1_pred_0",
        "true_0_pred_1",
    ]:
        group = dataframe[dataframe["case_group"] == group_name]
        if not group.empty:
            selected_parts.append(group.head(3 if group_name.endswith("_1") else 2))

    if selected_parts:
        selected = pd.concat(selected_parts, ignore_index=False)
    else:
        selected = dataframe.head(case_count)

    if len(selected) < case_count:
        remaining = dataframe.drop(index=selected.index, errors="ignore")
        selected = pd.concat([selected, remaining.head(case_count - len(selected))])

    return selected.head(case_count).copy()


def build_cases(case_count: int = 10) -> pd.DataFrame:
    if VAL_PREDICTIONS_FILE.exists():
        predictions = pd.read_csv(VAL_PREDICTIONS_FILE)
    else:
        from src.config import VAL_FILE

        validation_data = pd.read_csv(VAL_FILE)
        predictions = predict_dataframe(validation_data)

    required_columns = [TEXT_COLUMN, LABEL_COLUMN, "prediction", "reason"]
    missing_columns = [column for column in required_columns if column not in predictions.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns for explainability cases: {missing_columns}")

    cases = _select_cases(predictions, case_count=case_count)
    cases["rule_signals"] = cases[TEXT_COLUMN].map(_active_signal_names)
    cases["brief_analysis"] = cases.apply(_short_analysis, axis=1)

    output_columns = [
        column
        for column in [
            ID_COLUMN,
            TEXT_COLUMN,
            LABEL_COLUMN,
            "prediction",
            "reason",
            "rule_signals",
            "brief_analysis",
            EVENT_COLUMN,
        ]
        if column in cases.columns
    ]

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    cases[output_columns].to_csv(EXPLAINABILITY_CASES_FILE, index=False)
    return cases[output_columns]


def main() -> None:
    cases = build_cases()
    print(f"Saved {len(cases)} explainability cases to: {EXPLAINABILITY_CASES_FILE}")


if __name__ == "__main__":
    main()
