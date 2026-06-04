from src.preprocess import clean_text

RUMOR_CUES = [
    "breaking",
    "shocking",
    "urgent",
    "confirmed",
    "they are hiding",
    "cover-up",
    "must see",
]

NON_RUMOR_CUES = [
    "report",
    "according to",
    "statement",
    "announced",
    "official",
    "confirmed by",
]


def generate_explanation(text: str, label: int) -> str:
    cleaned_text = clean_text(text)

    rumor_hits = [cue for cue in RUMOR_CUES if cue in cleaned_text]
    non_rumor_hits = [cue for cue in cleaned_text.split() if cue in NON_RUMOR_CUES]

    if label == 1:
        if rumor_hits:
            return (
                "The text is classified as rumor because it uses urgent or emotionally charged "
                f"wording, including: {', '.join(rumor_hits[:3])}."
            )
        return (
            "The text is classified as rumor because its wording is assertive and lacks a clearly "
            "verifiable source in the text itself."
        )

    if non_rumor_hits:
        return (
            "The text is classified as non-rumor because it resembles informational reporting and "
            f"contains cues such as: {', '.join(non_rumor_hits[:3])}."
        )

    return (
        "The text is classified as non-rumor because it appears more descriptive than speculative "
        "and does not show strong rumor-like cues."
    )
