from __future__ import annotations

import argparse

from src.predict import predict_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict rumor label and generate explanation.")
    parser.add_argument("--text", required=True, help="Input text to classify")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label, reason = predict_text(args.text)

    print(f"Label: {label}")
    print(f"Reason: {reason}")


if __name__ == "__main__":
    main()
