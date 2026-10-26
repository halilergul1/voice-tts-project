"""Turkish text normalisation for TTS targets."""
from __future__ import annotations

import re
import unicodedata

from num2words import num2words

_MAP = {
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"', "‟": '"',
    "‘": "'", "’": "'", "‚": "'", "´": "'", "`": "'", "ʼ": "'",
    "–": "-", "—": "-", "‒": "-", "−": "-", "…": "...",
    " ": " ", "​": "",
}
ALLOWED_RE = re.compile(r"^[a-zA-ZçğıöşüÇĞİÖŞÜâîûÂÎÛ0-9 .,!?;:'\"\-]*$")
LOWER_MAP = str.maketrans({"I": "ı", "İ": "i"})


def basic_clean(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    for k, v in _MAP.items():
        t = t.replace(k, v)
    t = re.sub(r"\.{4,}", "...", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)
    t = re.sub(r"([.,!?;:])(?=[a-zA-ZçğıöşüÇĞİÖŞÜ])", r"\1 ", t)  # "evet.hayır" -> "evet. hayır"
    t = re.sub(r"\s+", " ", t).strip()
    return t


_NUM_WORDS_RE = re.compile(
    r"(trilyon|milyar|milyon|bin|yüz|doksan|seksen|yetmiş|altmış|elli|kırk|otuz|yirmi|on|dokuz|sekiz|yedi|altı|beş|dört|üç|iki|bir|sıfır)")


def _space_words(s: str) -> str:
    """num2words glues Turkish number words ('otuzyedibin'); Turkish orthography separates them."""
    return re.sub(r"\s+", " ", _NUM_WORDS_RE.sub(r" \1", s)).strip()


def _card(s: str) -> str:
    try:
        return _space_words(num2words(int(s), lang="tr"))
    except Exception:
        return s


def _ord(s: str) -> str:
    try:
        return _space_words(num2words(int(s), lang="tr", to="ordinal"))
    except Exception:
        return s


def _clock(m: re.Match) -> str:
    h, mi = int(m.group(1)), int(m.group(2))
    if h >= 24 or mi >= 60:
        return m.group(0)
    return _card(str(h)) if mi == 0 else f"{_card(str(h))} {_card(str(mi))}"


def expand_numbers(t: str) -> str:
    if not re.search(r"\d", t):
        return t
    t = re.sub(r"(\d)'(?=\w)", r"\1", t)                                   # 1923'te -> 1923te
    t = re.sub(r"%\s*(\d+)", lambda m: "yüzde " + _card(m.group(1)), t)     # %50
    t = re.sub(r"(\d+)\s*%", lambda m: "yüzde " + _card(m.group(1)), t)     # 50%
    t = re.sub(r"\b(\d{1,3})\.(\d{3})\b(?!\.\d)", r"\1\2", t)               # 1.000 -> 1000
    t = re.sub(r"\b(\d{1,2})[:.](\d{2})(?!\d|\.\d)", _clock, t)                # 14:30 / 08.00 / 14.30da (saat)
    t = re.sub(r"\b(\d+),(\d+)\b", lambda m: f"{_card(m.group(1))} virgül {_card(m.group(2))}", t)  # 3,5
    t = re.sub(r"\b(\d+)\.(?=\s+[a-zçğıöşü])", lambda m: _ord(m.group(1)), t)                       # 3. kat
    t = re.sub(r"\b(\d+)'?(inci|nci|ıncı|ncı|uncu|ncu|üncü|ncü)\b", lambda m: _ord(m.group(1)), t)  # 3'üncü
    t = re.sub(r"\d+", lambda m: _card(m.group(0)), t)
    return re.sub(r"\s+", " ", t).strip()


def normalize(text: str) -> str:
    return expand_numbers(basic_clean(text))


def charset_ok(text: str) -> bool:
    return bool(ALLOWED_RE.match(text))


def for_cer(text: str) -> str:
    """Lowercase (Turkish-aware), drop punctuation, expand digits, squeeze spaces."""
    t = expand_numbers(basic_clean(text))
    t = t.translate(LOWER_MAP).lower()
    t = t.replace("â", "a").replace("î", "i").replace("û", "u")
    t = re.sub(r"(?<=\w)'(?=\w)", "", t)  # Parvus'un -> parvusun (keep as one token)
    t = re.sub(r"[^a-zçğıöşü\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()
