import pytest

from ttsdata.textnorm import basic_clean, charset_ok, for_cer, normalize


@pytest.mark.parametrize("src,expected", [
    ("37655 dolar eksik.", "otuz yedi bin altı yüz elli beş dolar eksik."),
    ("Hâlen 400'e yakın çocuğumuz.", "Hâlen dört yüze yakın çocuğumuz."),
    ("3. kat, 14.30'da.", "üçüncü kat, on dört otuzda."),
    ("Saat 08.00'de gel.", "Saat sekizde gel."),
    ("%50 indirim", "yüzde elli indirim"),
    ("1923'te oldu.", "bin dokuz yüz yirmi üçte oldu."),
    ("3,5 lira", "üç virgül beş lira"),
    ("1.000 kişi", "bin kişi"),
    ("evet.hayır", "evet. hayır"),
    ("“Gel” dedi… ", '"Gel" dedi...'),
])
def test_normalize(src, expected):
    assert normalize(src) == expected


def test_charset():
    assert charset_ok(normalize("Kâğıt, hâlâ; 'Osmanlı' yıkılacak..."))
    assert not charset_ok("Merhaba ♪")
    assert not charset_ok("﻿Elbette")


def test_for_cer_turkish_lowercase_and_apostrophes():
    assert for_cer("Parvus'un yerini İSTANBUL'da bul!") == "parvusun yerini istanbulda bul"
    assert for_cer("Hâlen 2 kişi") == "halen iki kişi"
