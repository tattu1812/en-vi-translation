
from src.evaluate import (
    calculate_bleu,
    calculate_bleu_by_length,
    evaluate,
)


def test_perfect_translation():
    references = [
        "Tôi thích học ngôn ngữ và dịch máy.",
        "Hôm nay chúng tôi đang xây dựng một mô hình.",
        "Đây là một bản dịch hoàn toàn chính xác.",
    ]

    result = calculate_bleu(
        references,
        references
    )

    assert abs(result["bleu"] - 100.0) < 1e-6
    assert result["num_sentences"] == 3
    assert "tok:13a" in result["signature"]

    print("PASS: Perfect translation BLEU = 100")


def test_length_groups():
    sources = [
        " ".join(["word"] * 10),
        " ".join(["word"] * 11),
        " ".join(["word"] * 20),
        " ".join(["word"] * 21),
        " ".join(["word"] * 30),
        " ".join(["word"] * 31),
    ]

    references = [
        "Đây là một câu dịch mẫu có đủ từ."
        for _ in sources
    ]

    predictions = references.copy()

    result = calculate_bleu_by_length(
        sources,
        predictions,
        references
    )

    assert result["le_10"]["num_sentences"] == 1
    assert result["11_20"]["num_sentences"] == 2
    assert result["21_30"]["num_sentences"] == 2
    assert result["gt_30"]["num_sentences"] == 1

    print("PASS: Source length grouping")


def test_mismatched_counts():
    try:
        evaluate(
            ["Hello", "Good morning"],
            ["Xin chào"],
            ["Xin chào", "Chào buổi sáng"]
        )
    except ValueError:
        print("PASS: Mismatched counts rejected")
    else:
        raise AssertionError(
            "Expected ValueError"
        )


if __name__ == "__main__":
    test_perfect_translation()
    test_length_groups()
    test_mismatched_counts()

    print("\nALL BLEU TESTS PASSED")
