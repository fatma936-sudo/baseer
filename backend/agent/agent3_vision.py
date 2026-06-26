"""
Agent 3 — Vision / perception.  Model: Fanar-Oryx-IVU-2.

Reads the printed labels on the products to (a) report WHICH items are present
(describe_scene, grounded to the catalog) and (b) LOCALIZE them as pixel boxes
(locate_scene) — the box center drives the arm's pre-positioning before the grasp.
"""
import base64
import json
import re

import requests

from agent.fanar_base import FANAR_BASE_URL, VISION_MODEL, FanarError, _AUTH


def describe_scene(image_bytes, catalog):
    """Fanar-Oryx vision: return which catalog items are present.
    `catalog` can be a list of names, OR a dict {name: description} (registry) so it
    matches by brand text OR color/shape — and returns YOUR category names."""
    b64 = base64.b64encode(image_bytes).decode()
    if isinstance(catalog, dict):
        names = list(catalog.keys())
        lines = "\n".join(f"- {k}: {v}" for k, v in catalog.items())
        prompt = (
            "هذه صورة لطاولة زينة عليها منتجات تجميل وعناية. هذه قائمة المنتجات المحتملة، "
            "كل اسم متبوع بوصفه (الماركة واللون والشكل):\n" + lines +
            "\nطابِق كل منتج ظاهر في الصورة مع الاسم المناسب من القائمة، بالاعتماد على الملصق "
            "أو على اللون/الشكل إذا كان الملصق غير واضح. "
            'أعد JSON فقط بالأسماء العربية من القائمة: {"items":["..."]}. '
            "لا تكرر اسماً ولا تضف اسماً خارج القائمة."
        )
    else:
        names = list(catalog)
        prompt = (
            "هذه صورة لطاولة زينة عليها منتجات تجميل وعناية. من هذه القائمة المحددة فقط، "
            "أعد العناصر التي تظهر فعلاً في الصورة بالاعتماد على قراءة الملصقات: "
            + "، ".join(names)
            + '. أعد JSON فقط بنفس الأسماء: {"items":["..."]}. '
            "لا تضف أي عنصر غير موجود في القائمة أو غير ظاهر."
        )
    payload = {
        "model": VISION_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        "temperature": 0.1,
    }
    r = requests.post(
        f"{FANAR_BASE_URL.rstrip('/')}/chat/completions",
        headers={**_AUTH, "Content-Type": "application/json"}, json=payload, timeout=120)
    if r.status_code != 200:
        raise FanarError(f"VISION HTTP {r.status_code}: {r.text[:200]}")
    content = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", content, re.S)
    found = json.loads(m.group(0)).get("items", []) if m else []
    return [c for c in names if c in found]  # keep catalog order, drop anything off-list


def locate_scene(image_bytes, width, height, products=None):
    """Fanar-Oryx: return [{label, box:[x1,y1,x2,y2]}] for each product (localizes).
    If `products` (registry dict) is given, labels are mapped to YOUR category names
    (by brand OR color); otherwise it returns the raw printed label. For the live view."""
    b64 = base64.b64encode(image_bytes).decode()
    if products:
        lines = "\n".join(f"- {k}: {v}" for k, v in products.items())
        prompt = (
            f"الصورة عرضها {width} وارتفاعها {height} بكسل. هذه المنتجات المحتملة (الاسم: الوصف):\n"
            + lines +
            "\nلكل منتج ظاهر، طابِقه مع الاسم المناسب من القائمة (بالملصق أو اللون/الشكل) وأعد إطار "
            'إحداثيات بالبكسل. أعد JSON فقط: {"items":[{"label":"<اسم من القائمة>","box":[x1,y1,x2,y2]}]}'
        )
    else:
        prompt = (
            f"الصورة عرضها {width} وارتفاعها {height} بكسل. حدّد منتجات التجميل/العناية الظاهرة فقط. "
            "لكل منتج أعد النص المكتوب على ملصقه (الماركة/الرائحة كما هو مكتوب) وإطار إحداثيات بالبكسل. "
            'أعد JSON فقط: {"items":[{"label":"...","box":[x1,y1,x2,y2]}]}'
        )
    payload = {
        "model": VISION_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        "temperature": 0.1,
    }
    r = requests.post(
        f"{FANAR_BASE_URL.rstrip('/')}/chat/completions",
        headers={**_AUTH, "Content-Type": "application/json"}, json=payload, timeout=120)
    if r.status_code != 200:
        raise FanarError(f"VISION HTTP {r.status_code}: {r.text[:200]}")
    content = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"(\[.*\]|\{.*\})", content, re.S)
    if not m:
        return []
    data = json.loads(m.group(1))
    items = data["items"] if isinstance(data, dict) and "items" in data else (data if isinstance(data, list) else [])
    out = []
    for it in items:
        if isinstance(it, dict) and "box" in it and len(it["box"]) == 4:
            out.append({"label": str(it.get("label", "")), "box": [int(v) for v in it["box"]]})
    return out


def locate_one(image_bytes, width, height, name, description, samples=3):
    """FOCUSED, repeated localization of ONE object — more reliable + precise than the
    all-items query for driving the arm. Asks Oryx for just `name`'s tight box, `samples`
    times, and returns the MEDIAN center (u, v, n_hits) to smooth VLM variance. None if
    never found."""
    import statistics
    b64 = base64.b64encode(image_bytes).decode()
    prompt = (
        f"الصورة عرضها {width} وارتفاعها {height} بكسل. "
        f"جد فقط هذا الغرض: «{name}» — {description}. "
        "أعِد إطاراً محيطياً ضيّقاً ودقيقاً متمركزاً تماماً على الغرض نفسه (وليس على ما حوله). "
        'أعد JSON فقط بالشكل {"box":[x1,y1,x2,y2]} ، وإذا لم يظهر الغرض فأعد {"box":null}.'
    )
    payload = {
        "model": VISION_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        "temperature": 0.1,
    }
    centers = []
    for _ in range(samples):
        try:
            r = requests.post(f"{FANAR_BASE_URL.rstrip('/')}/chat/completions",
                              headers={**_AUTH, "Content-Type": "application/json"},
                              json=payload, timeout=120)
            if r.status_code != 200:
                continue
            content = r.json()["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", content, re.S)
            if not m:
                continue
            box = json.loads(m.group(0)).get("box")
            if box and len(box) == 4:
                x1, y1, x2, y2 = [float(v) for v in box]
                centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        except Exception:
            continue
    if not centers:
        return None
    u = statistics.median(c[0] for c in centers)
    v = statistics.median(c[1] for c in centers)
    return (u, v, len(centers))
