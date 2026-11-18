import unicodedata

from src.data.text_normalization import normalize_text


def test_unicode_nfc():
    expected = "Tôi học tiếng Việt."
    decomposed = unicodedata.normalize("NFD", expected)
    assert decomposed != expected
    assert normalize_text(decomposed, "vi") == expected


def test_punctuation_and_html_entities():
    assert normalize_text(
        "She said , &quot; Hello ! &quot;", "en"
    ) == 'She said, "Hello!"'
    assert normalize_text(
        "Cô ấy nói , &quot; Xin chào ! &quot;", "vi"
    ) == 'Cô ấy nói, "Xin chào!"'


def test_whitespace_and_case():
    assert normalize_text(" \tNASA\u00a0  and\nViệt Nam  ", "en") == "NASA and Việt Nam"
    assert normalize_text("  Ở trường ,   chúng tôi học tiếng Việt . ", "vi") == (
        "Ở trường, chúng tôi học tiếng Việt."
    )


def test_empty_text():
    for lang in ("en", "vi"):
        for text in ("", " \t\n\u00a0 ", "&nbsp;"):
            assert normalize_text(text, lang) == ""


def test_unsupported_language():
    try:
        normalize_text("Bonjour", "fr")
    except ValueError as error:
        assert str(error) == "Unsupported language: fr"
    else:
        raise AssertionError("Expected ValueError for unsupported language")


if __name__ == "__main__":
    test_unicode_nfc()
    test_punctuation_and_html_entities()
    test_whitespace_and_case()
    test_empty_text()
    test_unsupported_language()
    print("ALL TEXT NORMALIZATION TESTS PASSED")
