#!/usr/bin/env python3
"""Small end-to-end Clanker-LM demonstration."""

from clanker_lm import ClankerLM


def main() -> None:
    transcript = (
        "My sister bought a used Honda yesterday.",
        "Who bought the Honda?",
        "What did she buy?",
        "When did she buy it?",
        "Why did she buy it?",
        "She bought it because her old car broke down.",
        "Why did she buy it?",
    )

    with ClankerLM(session_id="example", strict_clanker=True) as dialogue:
        for message in transcript:
            result = dialogue.process(message)
            print(f"YOU: {message}")
            print(f"CLANKER-LM: {result.response}")
            if result.answer is not None:
                print(
                    "  contract:",
                    result.answer.status.value,
                    "certainty=", result.answer.certainty,
                    "source=", result.answer.provenance.value,
                )
            print()


if __name__ == "__main__":
    main()
