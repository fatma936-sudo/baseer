"""
ASR post-correction: Aura returns diacritized text and sometimes fuses words
(e.g. "عطر ديور" -> "عِطراديور"), which can derail the LLM. We strip diacritics
and map common ASR manglings / dialect synonyms back to the catalog names BEFORE
the agent reasons.
"""
import re

# Arabic harakat/tashkeel + tatweel
_TASHKEEL = re.compile(r"[ؗ-ًؚ-ْٰـ]")


def strip_diacritics(s):
    return _TASHKEEL.sub("", s or "")


# canonical catalog name  <-  variants (manglings + dialect synonyms)
_ALIASES = [
    (["عطراديور", "عطرديور", "عطر ديور", "البرفان", "برفان", "بارفان", "العطر", "عطر", "ديور"], "عطر ديور"),
    (["الكريم المرطب", "كريم مرطب", "المرطب", "الكريم", "كريم", "مرطب"], "كريم مرطب"),
    (["واقي الشمس", "واقي شمس", "الواقي", "صن بلوك", "صنبلوك", "واقي"], "واقي شمس"),
]


def normalize_command(text):
    """Strip diacritics and canonicalize any item mentions."""
    out = strip_diacritics(text)
    for variants, canon in _ALIASES:
        for v in sorted(variants, key=len, reverse=True):
            if v in out:
                out = out.replace(v, canon)
                break
    return out.strip()
