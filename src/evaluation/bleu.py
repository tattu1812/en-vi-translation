"""Compatibility entry point for the SacreBLEU evaluator."""

from src.evaluate import (
    DATA_DIR,
    LENGTH_GROUPS,
    calculate_bleu,
    calculate_bleu_by_length,
    evaluate,
    load_lines,
    main,
)


if __name__ == "__main__":
    main()
