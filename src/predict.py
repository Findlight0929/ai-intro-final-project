from __future__ import annotations

import joblib
import pandas as pd

from src.config import MODEL_FILE, TEXT_COLUMN
from src.preprocess import clean_text


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}. Run src/train_baseline.py first."
        )
    return joblib.load(MODEL_FILE)


def predict_text(text: str) -> int:
    model = load_model()
    cleaned_text = clean_text(text)
    prediction = model.predict([cleaned_text])[0]
    return int(prediction)


def predict_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    model = load_model()
    result = dataframe.copy()
    result[TEXT_COLUMN] = result[TEXT_COLUMN].fillna("").map(clean_text)
    result["prediction"] = model.predict(result[TEXT_COLUMN])
    return result
