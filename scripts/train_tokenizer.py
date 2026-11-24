"""Run the train_tokenizer pipeline from the repository root."""

if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.train_tokenizer import main


if __name__ == "__main__":
    main()
