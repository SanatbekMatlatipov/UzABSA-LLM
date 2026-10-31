#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Uzbek Cyrillic -> Latin transliteration
# =============================================================================
"""
The training data is Uzbek in the Latin script, but some reviews are written
in Uzbek Cyrillic, which the model has almost never seen. This module tells
Uzbek Cyrillic apart from Russian (Russian is left as it is) and
transliterates it to the Latin orthography of the training data, keeping a
character map so that aspect terms found in the Latin text can be mapped back
to the original review.

    >>> detect_script("Овқатлари жуда мазали, хизмат ҳам зўр")
    'uz_cyrl'
    >>> to_latin("Овқатлари жуда мазали")[0]
    'Ovqatlari juda mazali'
    >>> span_in_original("Ovqatlari", *to_latin("Овқатлари жуда мазали"), "Овқатлари жуда мазали")
    'Овқатлари'

Rules: the official Uzbek Latin alphabet (1995), written with the ASCII
apostrophe that the training data uses for o' / g' and the tutuq belgisi.
'е' is 'ye' at the start of a word or after a vowel and 'e' elsewhere; 'ц' is
'ts' after a vowel and 's' elsewhere; 'сҳ' is "s'h" so it is not read as 'sh'.

Author: UzABSA Team
License: MIT
"""

import re

CYR2LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ё": "yo", "ж": "j", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p",
    "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "x", "ч": "ch", "ш": "sh",
    "щ": "sh", "ъ": "'", "ь": "", "ы": "i", "э": "e", "ю": "yu", "я": "ya",
    "ў": "o'", "қ": "q", "ғ": "g'", "ҳ": "h",
    # look-alike letters from other Cyrillic keyboards
    "һ": "h", "ҡ": "q", "ӯ": "o'", "і": "i", "ҷ": "j",
}
VOWELS = set("аеёиоуэюяўы")
UZ_LETTERS = set("ўқғҳҷ")

# Frequent words of one language only (lower case; informal spellings included).
# Russian adjectives such as "свежий" or "сочный" are common inside Uzbek
# reviews, so the Russian side relies on function words and verb endings.
UZ_WORDS = set("""
ва бу шу жуда жудаям яхши яхшиям йахши ёмон учун билан бн эмас бор йўқ йук йок юк юқ
керак ҳам хам лекин аммо энг енг мен менга манга сиз сизга биз бизга бизага улар нарса
нарсалар кўп куп раҳмат рахмат зўр зур зор зури зўри ёқди ёкди ёкти ёқмади ёкмади
бўлди булди бўлса булса экан эди қилиш килиш қилди килди берди олдим олди келди
ишлайди ишламайди ўзи узи ҳамма хамма барча жой жойи хизмат хизмати хизматлари нархи
нархлар нархиям сифати сифат сифатли мазали маззали мазза ширин тез тезда секин яна ҳеч
хеч катта кичик дўкон докон дукон ходимлар ходимлари тавсия маъқул маъкул ажойиб аъло
ало ҳаммага хаммага ҳаммаси хаммаси овқат овкат овқатлари овкатлари таом таомлар
таомлари ичимлик пул нарх арзон қиммат киммат кимат ҳозир хозир кейин олдин бир икки
нега қандай кандай қачон качон ҳали хали фақат факат доим доимо ерда ердаги буюртма
етказиб бериш ишлар ишлари шунака шунақа гап хуллас айникса айниқса чиройли ёкимли
ёқимли юмшок юмшоқ озгина сал анча мумкин ишлатаман осон қулай кулай туйимли тўйимли
""".split())
RU_WORDS = set("""
и в во не на что это очень все всё всем я мы вы они он она с со по как но за так же
был была было были есть нет мне меня для только уже когда ещё еще или бы до от при из
к о а их его её вам нас нам там тут здесь где чем тоже всегда спасибо хорошо отлично
можно нужно будет ничего очень если потом после который которые даже раз уже чтобы
""".split())
UZ_SUFFIX = re.compile(r"(лар(и|ни|га|да|дан|нинг|имиз|ингиз|им|инг|ини|ига|ида|идан|ям|иям)?"
                       r"|нинг|ингиз|имиз|япти|моқда|мокда|ишди|ибди|ган|иб|аям|иям)$")
RU_SUFFIX = re.compile(r"(ться|тся|ешь|ется|ются|ает|яет|ует|ит|ал|ала|али|ого|ему|ыми|ими)$")


def is_cyrillic(ch):
    return "Ѐ" <= ch <= "ӿ"


def detect_script(text):
    """'latin', 'uz_cyrl' (Uzbek in Cyrillic) or 'ru' (Russian).

    A text counts as Cyrillic when most of its letters are Cyrillic; Uzbek and
    Russian are then told apart by counting the words that look like one
    language only (Uzbek letters, frequent words and suffixes against Russian
    function words, verb endings and 'ы'/'щ'). Ties go to Russian, so in doubt
    the text is left unchanged.
    """
    letters = [c for c in text if c.isalpha()]
    cyr = sum(is_cyrillic(c) for c in letters)
    if not cyr or cyr < len(letters) / 2:
        return "latin"
    uz = ru = 0
    for w in re.findall(r"[Ѐ-ӿ]+", text.lower()):
        if UZ_LETTERS & set(w) or w in UZ_WORDS or UZ_SUFFIX.search(w):
            uz += 1
        elif w in RU_WORDS or RU_SUFFIX.search(w) or set("ыщ") & set(w):
            ru += 1
    return "uz_cyrl" if uz > ru else "ru"


def _render(lat, ch, prev, nxt):
    """Case of a transliterated letter: 'Sh' in a capitalised word, 'SH' in an all-caps one."""
    if not ch.isupper() or not lat:
        return lat
    if (nxt and nxt.isupper()) or (prev and prev.isupper()):
        return lat.upper()
    return lat[0].upper() + lat[1:]


def to_latin(text):
    """Transliterate the Cyrillic letters of `text`; other characters pass through.

    Returns (latin_text, offsets) where offsets[j] is the index in `text` of
    the character that produced latin_text[j].
    """
    out, offsets = [], []
    for i, ch in enumerate(text):
        low = ch.lower()
        prev = text[i - 1] if i else ""
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if low == "е":
            start = not prev.isalpha() or prev.lower() in VOWELS or prev.lower() in "ъь"
            lat = "ye" if start else "e"
        elif low == "ц":
            lat = "ts" if prev.lower() in VOWELS else "s"
        elif low == "с" and nxt.lower() == "ҳ":
            lat = "s'"
        elif low == "ь" and nxt.lower() == "о":
            lat = "y"
        elif low in CYR2LAT:
            lat = CYR2LAT[low]
        else:
            out.append(ch)
            offsets.append(i)
            continue
        lat = _render(lat, ch, prev, nxt)
        out.append(lat)
        offsets.extend([i] * len(lat))
    return "".join(out), offsets


def span_in_original(term, latin_text, offsets, original):
    """The substring of `original` that a term found in the Latin text came from.

    Returns None when the term does not occur in the Latin text, e.g. when the
    model normalised its spelling.
    """
    term = (term or "").strip()
    if not term:
        return None
    pos = latin_text.find(term)
    if pos < 0:
        pos = latin_text.lower().find(term.lower())
    if pos < 0:
        return None
    return original[offsets[pos]: offsets[pos + len(term) - 1] + 1]


def prepare(text):
    """What the model should see: (model_text, script, offsets or None)."""
    script = detect_script(text)
    if script != "uz_cyrl":
        return text, script, None
    latin, offsets = to_latin(text)
    return latin, script, offsets


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)
    for s in ["Ошхонада овқатлари жуда мазали, хизмат ҳам зўр. Раҳмат!",
              "Еттинчи марта келдим, ЦЕНТРдаги филиали яхши, Исҳоқ ака раҳмат",
              "Очень вкусно, персонал вежливый, рекомендую!"]:
        print(detect_script(s), "|", prepare(s)[0])
