from __future__ import annotations

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from src.config import (
    LABEL_COLUMN,
    MODEL_FILE,
    MODELS_DIR,
    OUTPUTS_DIR,
    TEXT_COLUMN,
    TRAIN_FILE,
    VAL_FILE,
    VAL_PREDICTIONS_FILE,
)
from src.preprocess import clean_text


def load_dataset(path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data[[TEXT_COLUMN, LABEL_COLUMN]].dropna().copy()
    data[TEXT_COLUMN] = data[TEXT_COLUMN].map(clean_text)
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
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(TRAIN_FILE)
    val_df = load_dataset(VAL_FILE)

    pipeline = build_pipeline()
    pipeline.fit(train_df[TEXT_COLUMN], train_df[LABEL_COLUMN])

    predictions = pipeline.predict(val_df[TEXT_COLUMN])
    accuracy = accuracy_score(val_df[LABEL_COLUMN], predictions)

    print(f"Validation accuracy: {accuracy:.4f}")
    print(classification_report(val_df[LABEL_COLUMN], predictions, digits=4))

    result_df = val_df.copy()
    result_df["prediction"] = predictions
    result_df.to_csv(VAL_PREDICTIONS_FILE, index=False)

    joblib.dump(pipeline, MODEL_FILE)
    print(f"Saved model to: {MODEL_FILE}")
    print(f"Saved validation predictions to: {VAL_PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()
