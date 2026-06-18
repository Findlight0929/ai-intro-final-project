from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from src.preprocess import clean_text
from src.rag import RetrievedExample, format_retrieved_examples, retrieve_relevant_examples


EXAGGERATION_CUES = [
    "breaking",
    "shocking",
    "urgent",
    "must see",
    "wake up",
    "unbelievable",
    "bombshell",
    "viral",
    "share this",
    "share before",
    "asap",
]

ABSOLUTE_CUES = [
    "always",
    "never",
    "everyone",
    "everybody",
    "no one",
    "all",
    "none",
    "only",
    "clearly",
    "definitely",
    "must",
    "cannot",
    "guaranteed",
    "100%",
]

SOURCE_CUES = [
    "according to",
    "official",
    "officials",
    "officials said",
    "police said",
    "statement",
    "announced",
    "reported",
    "reports",
    "confirmed",
    "confirmed by",
    "press release",
    "interview",
    "reuters",
    "associated press",
    "ap news",
    "bbc",
    "cnn",
]

SPECULATION_CUES = [
    "maybe",
    "might",
    "could",
    "seems",
    "apparently",
    "allegedly",
    "unconfirmed",
    "cannot confirm",
    "please check",
    "isn't that",
    "i heard",
    "rumor",
]

RUMOR_PATTERN_CUES = [
    "cover-up",
    "cover up",
    "hiding",
    "what are they hiding",
    "the truth",
    "exposed",
    "smear campaign",
    "obtained audio",
    "will release",
    "leaked",
    "secret",
    "lie",
    "lies",
    "hoax",
]

URL_PATTERN = re.compile(r"<url>|https?://\S+|www\.\S+")
HASHTAG_PATTERN = re.compile(r"#\w+")
QUESTION_PATTERN = re.compile(r"\?")


@dataclass(frozen=True)
class ExplanationSignal:
    name: str
    description: str
    hits: list[str]

    @property
    def has_hits(self) -> bool:
        return bool(self.hits)


@dataclass(frozen=True)
class ExplanationResult:
    label: int
    label_name: str
    reason: str
    signals: list[ExplanationSignal]
    retrieved_examples: list[RetrievedExample]
    llm_prompt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "label_name": self.label_name,
            "reason": self.reason,
            "signals": [
                {
                    "name": signal.name,
                    "description": signal.description,
                    "hits": signal.hits,
                }
                for signal in self.signals
            ],
            "retrieved_examples": [
                example.to_dict()
                for example in self.retrieved_examples
            ],
            "llm_prompt": self.llm_prompt,
        }


def _find_cues(text: str, cues: list[str]) -> list[str]:
    hits = []
    for cue in cues:
        if _cue_in_text(text, cue) and cue not in hits:
            hits.append(cue)
    return hits


def _cue_in_text(text: str, cue: str) -> bool:
    escaped_cue = re.escape(cue)
    starts_with_word = cue[0].isalnum()
    ends_with_word = cue[-1].isalnum()

    if starts_with_word and ends_with_word:
        pattern = rf"(?<![a-z0-9]){escaped_cue}(?![a-z0-9])"
        return re.search(pattern, text) is not None
    if starts_with_word:
        pattern = rf"(?<![a-z0-9]){escaped_cue}"
        return re.search(pattern, text) is not None
    if ends_with_word:
        pattern = rf"{escaped_cue}(?![a-z0-9])"
        return re.search(pattern, text) is not None
    return cue in text


def _format_hits(hits: list[str], max_hits: int = 3) -> str:
    return ", ".join(hits[:max_hits])


def analyze_text_signals(text: str) -> list[ExplanationSignal]:
    cleaned_text = clean_text(text)
    source_hits = _find_cues(cleaned_text, SOURCE_CUES)

    signals = [
        ExplanationSignal(
            name="exaggeration_or_incitement",
            description="contains exaggerated, urgent, or emotionally provocative wording",
            hits=_find_cues(cleaned_text, EXAGGERATION_CUES),
        ),
        ExplanationSignal(
            name="absolute_wording",
            description="uses absolute wording that may overstate certainty",
            hits=_find_cues(cleaned_text, ABSOLUTE_CUES),
        ),
        ExplanationSignal(
            name="missing_reliable_source",
            description="does not show a clearly verifiable source cue",
            hits=[] if source_hits else ["no explicit source cue"],
        ),
        ExplanationSignal(
            name="rumor_pattern",
            description="resembles common rumor patterns such as hiding, leaking, or cover-up claims",
            hits=_find_cues(cleaned_text, RUMOR_PATTERN_CUES),
        ),
        ExplanationSignal(
            name="speculation",
            description="uses uncertain or speculative wording",
            hits=_find_cues(cleaned_text, SPECULATION_CUES),
        ),
        ExplanationSignal(
            name="source_or_reporting",
            description="contains source, reporting, or confirmation cues",
            hits=source_hits,
        ),
        ExplanationSignal(
            name="social_media_context",
            description="contains social-media markers such as links or hashtags",
            hits=_social_media_hits(cleaned_text),
        ),
    ]
    return signals


def _social_media_hits(cleaned_text: str) -> list[str]:
    hits = []
    if URL_PATTERN.search(cleaned_text):
        hits.append("link")
    if HASHTAG_PATTERN.search(cleaned_text):
        hits.append("hashtag")
    if QUESTION_PATTERN.search(cleaned_text):
        hits.append("question")
    return hits


def _summarize_retrieval_context(examples: list[RetrievedExample], label: int) -> str:
    if not examples:
        return "RAG context: no sufficiently similar training example was retrieved."

    label = int(label)
    same_label_count = sum(1 for example in examples if example.label == label)
    total_count = len(examples)
    label_name = "rumor" if label == 1 else "non-rumor"

    if same_label_count == total_count:
        return (
            f"RAG context: the {total_count} nearest training examples are also labeled "
            f"{label_name}, so the prediction is close to similar learned cases."
        )
    if same_label_count > 0:
        return (
            f"RAG context: {same_label_count}/{total_count} nearest training examples share "
            f"the predicted {label_name} label, while the rest show mixed context."
        )
    return (
        "RAG context: the nearest training examples have different labels, "
        "so this explanation should be treated cautiously."
    )


def generate_llm_prompt(
    text: str,
    label: int,
    signals: list[ExplanationSignal] | None = None,
    retrieved_examples: list[RetrievedExample] | None = None,
) -> str:
    label_name = "rumor" if int(label) == 1 else "non-rumor"
    active_signals = signals or analyze_text_signals(text)
    active_examples = retrieved_examples if retrieved_examples is not None else retrieve_relevant_examples(text)
    signal_lines = [
        f"- {signal.name}: {', '.join(signal.hits)}"
        for signal in active_signals
        if signal.has_hits
    ]
    signal_block = "\n".join(signal_lines) if signal_lines else "- no obvious rule signal"
    retrieved_block = format_retrieved_examples(active_examples)

    return (
        "You are helping an explainable rumor-detection system.\n"
        "Task: write one concise explanation in English for the model prediction.\n"
        "Requirements:\n"
        "1. Do not claim the text is certainly true or false.\n"
        "2. Mention only observable textual cues.\n"
        "3. Use retrieved training examples only as dataset context, not as proof.\n"
        "4. Keep the explanation within 2 sentences.\n\n"
        f"Input text:\n{text}\n\n"
        f"Model prediction: {label_name} ({label})\n\n"
        f"Rule signals:\n{signal_block}\n\n"
        f"Retrieved training examples:\n{retrieved_block}\n\n"
        "Explanation:"
    )


def generate_rule_based_explanation(text: str, label: int) -> ExplanationResult:
    label = int(label)
    signals = analyze_text_signals(text)
    retrieved_examples = retrieve_relevant_examples(text)
    signal_map = {signal.name: signal for signal in signals}

    if label == 1:
        reasons = []
        for signal_name in [
            "exaggeration_or_incitement",
            "absolute_wording",
            "rumor_pattern",
            "speculation",
        ]:
            signal = signal_map[signal_name]
            if signal.has_hits:
                reasons.append(f"{signal.description}, such as {_format_hits(signal.hits)}")

        if signal_map["missing_reliable_source"].has_hits:
            reasons.append("does not provide an explicit source or confirmation cue")

        if not reasons:
            reasons.append("is closer to rumor-like textual patterns learned by the classifier")

        reason = "The text is classified as rumor because it " + "; and it ".join(reasons) + "."
        label_name = "rumor"
    else:
        reasons = []
        source_signal = signal_map["source_or_reporting"]
        if source_signal.has_hits:
            reasons.append(f"contains reporting or source-related cues such as {_format_hits(source_signal.hits)}")

        risky_signal_names = [
            "exaggeration_or_incitement",
            "absolute_wording",
            "rumor_pattern",
            "speculation",
        ]
        if not any(signal_map[name].has_hits for name in risky_signal_names):
            reasons.append("does not show strong urgent, absolute, speculative, or rumor-pattern wording")

        if signal_map["social_media_context"].has_hits:
            reasons.append(
                f"uses social-media markers ({_format_hits(signal_map['social_media_context'].hits)}) "
                "that should be checked against external evidence"
            )

        if not reasons:
            reasons.append("is closer to non-rumor textual patterns learned by the classifier")

        reason = "The text is classified as non-rumor because it " + "; and it ".join(reasons) + "."
        label_name = "non-rumor"

    reason = f"{reason} {_summarize_retrieval_context(retrieved_examples, label)}"

    return ExplanationResult(
        label=label,
        label_name=label_name,
        reason=reason,
        signals=signals,
        retrieved_examples=retrieved_examples,
        llm_prompt=generate_llm_prompt(text, label, signals, retrieved_examples),
    )


def generate_explanation(text: str, label: int) -> str:
    return generate_rule_based_explanation(text, label).reason


def generate_explanation_detail(text: str, label: int, use_llm: bool = False) -> ExplanationResult:
    result = generate_rule_based_explanation(text, label)
    if not use_llm:
        return result

    llm_reason = call_school_llm(result.llm_prompt)
    if not llm_reason:
        return result

    return ExplanationResult(
        label=result.label,
        label_name=result.label_name,
        reason=llm_reason,
        signals=result.signals,
        retrieved_examples=result.retrieved_examples,
        llm_prompt=result.llm_prompt,
    )


def call_school_llm(prompt: str) -> str | None:
    """Call an OpenAI-compatible school LLM endpoint when environment variables are set.

    Required environment variables:
    - SCHOOL_LLM_API_URL: chat-completions endpoint URL
    - SCHOOL_LLM_API_KEY: API key or token

    Optional environment variables:
    - SCHOOL_LLM_MODEL: model name, default "school-llm"
    - SCHOOL_LLM_TIMEOUT: timeout seconds, default "20"
    """
    api_url = os.getenv("SCHOOL_LLM_API_URL")
    api_key = os.getenv("SCHOOL_LLM_API_KEY")
    if not api_url or not api_key:
        return None

    payload = {
        "model": os.getenv("SCHOOL_LLM_MODEL", "school-llm"),
        "messages": [
            {
                "role": "system",
                "content": "You generate concise, faithful explanations for rumor-detection predictions.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        timeout = float(os.getenv("SCHOOL_LLM_TIMEOUT", "20"))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError):
        return None

    try:
        return response_data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a rule-based explanation for one text.")
    parser.add_argument("--text", required=True, help="Input text")
    parser.add_argument("--label", required=True, type=int, choices=[0, 1], help="Model prediction label")
    parser.add_argument("--detail", action="store_true", help="Print JSON details with rule signals")
    parser.add_argument("--use-llm", action="store_true", help="Use the school LLM endpoint if configured")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detail = generate_explanation_detail(args.text, args.label, use_llm=args.use_llm)
    if args.detail:
        print(json.dumps(detail.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(detail.reason)


if __name__ == "__main__":
    main()
