from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import EVENT_COLUMN, ID_COLUMN, LABEL_COLUMN, TEXT_COLUMN, TRAIN_FILE
from src.preprocess import clean_text


@dataclass(frozen=True)
class RetrievedExample:
    text: str
    label: int
    score: float
    event: Any | None = None
    example_id: Any | None = None

    @property
    def label_name(self) -> str:
        return "rumor" if self.label == 1 else "non-rumor"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "label_name": self.label_name,
            "score": self.score,
            "event": self.event,
            "id": self.example_id,
        }


@dataclass(frozen=True)
class RagIndex:
    dataframe: pd.DataFrame
    vectorizer: TfidfVectorizer
    matrix: Any


def _to_python_value(value: Any) -> Any | None:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


@lru_cache(maxsize=1)
def _load_rag_index() -> RagIndex | None:
    if not TRAIN_FILE.exists():
        return None

    dataframe = pd.read_csv(TRAIN_FILE)
    required_columns = [TEXT_COLUMN, LABEL_COLUMN]
    if any(column not in dataframe.columns for column in required_columns):
        return None

    dataframe = dataframe.dropna(subset=required_columns).copy()
    if dataframe.empty:
        return None

    dataframe["clean_text"] = dataframe[TEXT_COLUMN].map(clean_text)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(dataframe["clean_text"])
    return RagIndex(dataframe=dataframe, vectorizer=vectorizer, matrix=matrix)


def retrieve_relevant_examples(
    text: str,
    top_k: int = 3,
    min_score: float = 0.05,
) -> list[RetrievedExample]:
    """Retrieve similar training examples for explanation-time RAG context."""
    index = _load_rag_index()
    if index is None or top_k <= 0:
        return []

    query_vector = index.vectorizer.transform([clean_text(text)])
    scores = (index.matrix @ query_vector.T).toarray().ravel()
    if scores.size == 0:
        return []

    top_indices = scores.argsort()[::-1][:top_k]
    examples = []
    for row_index in top_indices:
        score = float(scores[row_index])
        if score < min_score:
            continue

        row = index.dataframe.iloc[int(row_index)]
        examples.append(
            RetrievedExample(
                text=str(row[TEXT_COLUMN]),
                label=int(row[LABEL_COLUMN]),
                score=score,
                event=_to_python_value(row[EVENT_COLUMN]) if EVENT_COLUMN in row else None,
                example_id=_to_python_value(row[ID_COLUMN]) if ID_COLUMN in row else None,
            )
        )
    return examples


def format_retrieved_examples(
    examples: list[RetrievedExample],
    max_text_length: int = 180,
) -> str:
    if not examples:
        return "- no similar training example retrieved"

    lines = []
    for index, example in enumerate(examples, start=1):
        text = " ".join(example.text.split())
        if len(text) > max_text_length:
            text = text[: max_text_length - 3].rstrip() + "..."
        lines.append(
            f"- Example {index}: label={example.label_name} ({example.label}), "
            f"similarity={example.score:.3f}, text=\"{text}\""
        )
    return "\n".join(lines)
