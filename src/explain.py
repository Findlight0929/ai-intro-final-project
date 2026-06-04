from __future__ import annotations

from src.preprocess import clean_text

RUMOR_CUES = [
    "breaking",
    "shocking",
    "urgent",
    "cover-up",
    "must see",
    "cannot confirm",
    "unconfirmed",
    "rumor",
    "hiding",
    "lie",
    "lies",
]

SOURCE_CUES = [
    "according to",
    "official",
    "statement",
    "announced",
    "report",
    "reports",
    "confirmed by",
    "press release",
    "interview",
]

SPECULATION_CUES = [
    "maybe",
    "might",
    "could",
    "seems",
    "apparently",
    "cannot confirm",
    "please check",
    "isn't that",
]


def _find_cues(text: str, cues: list[str]) -> list[str]:
    return [cue for cue in cues if cue in text]


def generate_explanation(text: str, label: int) -> str:
    cleaned_text = clean_text(text)
    rumor_hits = _find_cues(cleaned_text, RUMOR_CUES)
    source_hits = _find_cues(cleaned_text, SOURCE_CUES)
    speculation_hits = _find_cues(cleaned_text, SPECULATION_CUES)

    if label == 1:
        reasons = []
        if rumor_hits:
            reasons.append(f"contains rumor-related cues such as {', '.join(rumor_hits[:3])}")
        if speculation_hits:
            reasons.append(f"uses uncertain or speculative wording such as {', '.join(speculation_hits[:3])}")
        if not source_hits:
            reasons.append("does not provide a clearly verifiable source in the text")
        if not reasons:
            reasons.append("matches the learned textual patterns of rumor samples in the training data")
        return "The text is classified as rumor because it " + "; and it ".join(reasons) + "."

    reasons = []
    if source_hits:
        reasons.append(f"contains reporting or source-related cues such as {', '.join(source_hits[:3])}")
    if not rumor_hits and not speculation_hits:
        reasons.append("does not show strong urgent, speculative, or rumor-like wording")
    if not reasons:
        reasons.append("is closer to the learned non-rumor patterns in the training data")
    return "The text is classified as non-rumor because it " + "; and it ".join(reasons) + "."
