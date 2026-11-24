"""Run the clean_iwslt15 pipeline from the repository root."""

if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clean_iwslt15 import main


if __name__ == "__main__":
    main()
