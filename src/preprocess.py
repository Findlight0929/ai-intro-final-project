import html
import re

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    normalized = html.unescape(text)
    normalized = normalized.strip().lower()
    normalized = URL_PATTERN.sub(" <url> ", normalized)
    normalized = MENTION_PATTERN.sub(" <user> ", normalized)
    normalized = WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()
