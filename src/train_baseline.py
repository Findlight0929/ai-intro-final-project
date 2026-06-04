from __future__ import annotations

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from src.config import (
    EXAMPLES_FILE,
    ID_COLUMN,
    LABEL_COLUMN,
    METRICS_FILE,
    MODEL_FILE,
    MODELS_DIR,
    OUTPUTS_DIR,
    TEXT_COLUMN,
    TRAIN_FILE,
    VAL_FILE,
    VAL_PREDICTIONS_FILE,
)
from src.explain import generate_explanation
from src.preprocess import clean_text


def load_dataset(path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required_columns = [TEXT_COLUMN, LABEL_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {path}: {missing_columns}")

    data = data.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()
    data["clean_text"] = data[TEXT_COLUMN].map(clean_text)
    return data


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def save_outputs(val_df: pd.DataFrame, predictions, report: str, accuracy: float) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    result_df = val_df.copy()
    result_df["prediction"] = predictions
    result_df["reason"] = [
        generate_explanation(text, int(label))
        for text, label in zip(result_df[TEXT_COLUMN], result_df["prediction"])
    ]

    output_columns = [column for column in [ID_COLUMN, TEXT_COLUMN, LABEL_COLUMN, "prediction", "reason"] if column in result_df.columns]
    result_df[output_columns].to_csv(VAL_PREDICTIONS_FILE, index=False)
    result_df[output_columns].head(20).to_csv(EXAMPLES_FILE, index=False)

    METRICS_FILE.write_text(
        f"Validation accuracy: {accuracy:.4f}\n\n{report}\n",
        encoding="utf-8",
    )


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(TRAIN_FILE)
    val_df = load_dataset(VAL_FILE)

    pipeline = build_pipeline()
    pipeline.fit(train_df["clean_text"], train_df[LABEL_COLUMN])

    predictions = pipeline.predict(val_df["clean_text"])
    accuracy = accuracy_score(val_df[LABEL_COLUMN], predictions)
    report = classification_report(val_df[LABEL_COLUMN], predictions, digits=4)

    print(f"Validation accuracy: {accuracy:.4f}")
    print(report)

    joblib.dump(pipeline, MODEL_FILE)
    save_outputs(val_df, predictions, report, accuracy)

    print(f"Saved model to: {MODEL_FILE}")
    print(f"Saved validation predictions to: {VAL_PREDICTIONS_FILE}")
    print(f"Saved example explanations to: {EXAMPLES_FILE}")
    print(f"Saved metrics to: {METRICS_FILE}")


if __name__ == "__main__":
    main()
