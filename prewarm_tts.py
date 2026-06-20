"""
Pre-generate the FIXED spoken phrases into tts_cache/ so they never hit the
Aura rate limit again. Run ONCE after your Fanar quota resets:

    /opt/anaconda3/envs/lerobot/bin/python prewarm_tts.py
"""
import os
import hashlib

from fanar import synthesize, FanarError

TTS_CACHE = os.path.join(os.path.dirname(__file__), "tts_cache")
os.makedirs(TTS_CACHE, exist_ok=True)

# Must match the /tts cache key in server.py: md5(voice + "|" + text), voice=""
PHRASES = [
    "أهلاً بك في بَصير، مساعدك الصوتي. النظام جاهز. لطلب غرض، المس الشاشة مع الاستمرار، قل ما تريد، ثم ارفع إصبعك.",
    "تم استلام الأمر، قيد التنفيذ",
    "عذراً، صار خطأ",
    "تم.",
]

for t in PHRASES:
    key = hashlib.md5(("|" + t).encode("utf-8")).hexdigest()
    path = os.path.join(TTS_CACHE, key + ".mp3")
    if os.path.exists(path):
        print("already cached:", t[:30])
        continue
    try:
        with open(path, "wb") as f:
            f.write(synthesize(t))
        print("cached OK    :", t[:30])
    except FanarError as e:
        print("FAILED       :", t[:30], "->", str(e)[:80])
