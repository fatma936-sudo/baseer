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


# canonical catalog name  <-  variants (ASR manglings + dialect synonyms).
# NOTE: bare "سيروم"/"السيروم" is intentionally NOT mapped — it must stay ambiguous
# (two serums exist) so the agent asks which one. Only specific serums map.
_ALIASES = [
    (["عطر شيا", "شيا", "shea"], "عطر شيا"),
    (["عطر المسك الأبيض", "المسك الأبيض", "المسك الابيض", "المسك", "مسك", "white musk", "musk"], "عطر المسك الأبيض"),
    (["عطر الياسمين", "الياسمين", "ياسمين", "عطر ياسمين", "jasmine"], "عطر الياسمين"),
    (["شامبو جاف", "الشامبو الجاف", "شامبو", "dry shampoo", "ليفينغ بروف", "living proof"], "شامبو جاف"),
    (["بودرة الشعر", "بودرة تصفيف الشعر", "البودرة", "بودرة", "باودر", "powder spray", "styling powder"], "بودرة الشعر"),
    (["سيروم الشعر", "سيروم شعر", "hair serum", "كيراستاز", "kerastase"], "سيروم الشعر"),
    (["سيروم الوجه", "سيروم الوجة", "السيروم الليلي", "سيروم ليلي", "فيشي", "نورمادرم", "normaderm", "double correction", "face serum", "night serum"], "سيروم الوجه"),
]


# ASR mishearings to correct BEFORE matching. e.g. Aura hears "سيروم" as "سيرة"
# (biography) — which can even trip Fanar's safety filter. Fix it back.
_ASR_FIX = {"السيرة": "السيروم", "سيرة": "سيروم", "السيره": "السيروم", "سيره": "سيروم"}


def normalize_command(text):
    """Strip diacritics, fix known ASR mishearings, and canonicalize item mentions."""
    out = strip_diacritics(text)
    for wrong, right in _ASR_FIX.items():
        out = out.replace(wrong, right)
    for variants, canon in _ALIASES:
        for v in sorted(variants, key=len, reverse=True):
            if v in out:
                out = out.replace(v, canon)
                break
    return out.strip()
