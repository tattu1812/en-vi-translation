
import html
import re
import unicodedata

from sacremoses import MosesDetokenizer

DETOKENIZERS = {
    "en": MosesDetokenizer(lang="en"),
    "vi": MosesDetokenizer(lang="vi"),
}


def normalize_text(text: str, lang: str) -> str:
    """
    Standardize tokenized IWSLT'15 text.

    - Unicode NFC
    - HTML unescaping
    - Moses detokenization
    - Whitespace normalization
    - Preserve original case and meaning
    """
    if lang not in DETOKENIZERS:
        raise ValueError(f"Unsupported language: {lang}")

    # 1. Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Decode HTML entities
    text = html.unescape(text)

    # 3. Normalize whitespace
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    # 4. Restore punctuation spacing
    # Input is expected to be tokenized IWSLT'15 text.
    text = DETOKENIZERS[lang].detokenize(
        text.split(),
        return_str=True,
        unescape=False,
    )

    # 5. Normalize whitespace after detokenization
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Preserve Unicode NFC
    return unicodedata.normalize("NFC", text)
