from __future__ import annotations

import argparse

import joblib
import pandas as pd

from src.config import MODEL_FILE, TEXT_COLUMN
from src.explain import generate_explanation
from src.preprocess import clean_text


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}. Run `python -m src.train_baseline` first."
        )
    return joblib.load(MODEL_FILE)


def predict_text(text: str) -> tuple[int, str]:
    model = load_model()
    cleaned_text = clean_text(text)
    label = int(model.predict([cleaned_text])[0])
    reason = generate_explanation(text, label)
    return label, reason


def predict_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    model = load_model()
    result = dataframe.copy()
    result[TEXT_COLUMN] = result[TEXT_COLUMN].fillna("")
    cleaned_texts = result[TEXT_COLUMN].map(clean_text)
    result["prediction"] = model.predict(cleaned_texts)
    result["reason"] = [
        generate_explanation(text, int(label))
        for text, label in zip(result[TEXT_COLUMN], result["prediction"])
    ]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict rumor labels for one CSV file.")
    parser.add_argument("--input", required=True, help="Input CSV path with a text column")
    parser.add_argument("--output", required=True, help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input)
    result = predict_dataframe(data)
    result.to_csv(args.output, index=False)
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
