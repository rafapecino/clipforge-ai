"""Emojis contextuales renderizados como PNG (Noto Emoji) con animación pop.

Se colocan por: (1) citas exactas sugeridas por el LLM, o (2) mapa de
palabras clave. Espaciados ≥2.5 s y limitados por clip para no saturar.
"""
import re
import unicodedata
from pathlib import Path
import cv2
import numpy as np

MAPA = [
    (r"dinero|millon|euro|pasta|caro|precio", "💰"),
    (r"\bjaja|\brisa|gracioso", "😂"),
    (r"incre[ií]ble|brutal|locura|tremend|bestial|flip", "🔥"),
    (r"\bojo\b|cuidado|atenci[oó]n", "👀"),
    (r"ca[ií]d|al suelo|highside|lowside", "💥"),
    (r"golpe|bomba|explot|estall", "💥"),
    (r"fichaje|renovaci|contrato|rumor", "✍️"),
    (r"error|fallo|desastre|fatal", "❌"),
    (r"\bgan(a|ó|ar)|campe[oó]n|victoria|triunf", "🏆"),
    (r"carrera|circuito|adelanta|\bmoto", "🏍️"),
    (r"r[aá]pid|velocidad|volando", "⚡"),
    (r"verg[uü]enza|enfad|cabre|indignante", "😡"),
    (r"no puede ser|alucin|sorprend|impact", "😱"),
    (r"peligro|riesgo|amenaza", "⚠️"),
    (r"sanci[oó]n|multa|castigo|il[eí]gal", "🚨"),
    (r"objetivo|clave|exacto|justo", "🎯"),
    (r"sube|crece|r[eé]cord|m[aá]ximo", "📈"),
    (r"explota la cabeza|flipas|mental", "🤯"),
]


def _norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def _codigos(e, sep, con_fe0f=False):
    return sep.join(f"{ord(c):x}" for c in e if con_fe0f or ord(c) != 0xFE0F)


def descargar(e, carpeta):
    """PNG BGRA 220 px del emoji (Noto → Twemoji de respaldo), con caché."""
    carpeta = Path(carpeta)
    destino = carpeta / f"e_{_codigos(e, '_')}.png"
    if destino.exists():
        return str(destino)
    import requests
    urls = [
        f"https://raw.githubusercontent.com/googlefonts/noto-emoji/main/png/512/emoji_u{_codigos(e, '_')}.png",
        f"https://cdn.jsdelivr.net/gh/jdecked/twemoji@15.1.0/assets/72x72/{_codigos(e, '-')}.png",
        f"https://cdn.jsdelivr.net/gh/jdecked/twemoji@15.1.0/assets/72x72/{_codigos(e, '-', True)}.png",
    ]
    for u in urls:
        try:
            r = requests.get(u, timeout=15)
            if r.status_code != 200:
                continue
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None or img.ndim != 3:
                continue
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img = cv2.resize(img, (220, 220), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(str(destino), img)
            return str(destino)
        except Exception:
            continue
    return None


def momentos(clip, palabras_out, carpeta, activo=True, maximo=6):
    """[(t_salida, ruta_png)] para un clip."""
    if not activo or not palabras_out:
        return []
    brutos = []
    for m in clip.get("emojis", []) or []:      # citas del LLM
        cita, emo = str(m.get("quote", "")), str(m.get("emoji", "")).strip()
        if not cita or not emo:
            continue
        primera = _norm(cita.split()[0]) if cita.split() else ""
        for w in palabras_out:
            if primera and _norm(w["w"]) == primera:
                brutos.append((w["s"], emo))
                break
    if not brutos:                              # respaldo: mapa de keywords
        for w in palabras_out:
            bajo = w["w"].lower()
            for rx, emo in MAPA:
                if re.search(rx, bajo):
                    brutos.append((w["s"], emo))
                    break
    brutos.sort()
    out, previo = [], -9.0
    for t, e in brutos:
        if t < 1.0 or t - previo < 2.5:
            continue
        png = descargar(e, carpeta)
        if png:
            out.append((float(t), png))
            previo = t
        if len(out) >= maximo:
            break
    return out


def _ease_out_back(p):
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (p - 1) ** 3 + c1 * (p - 1) ** 2


def pegar(frame, png_bgra, cx, cy, escala=1.0, alfa=1.0):
    """Pega un PNG BGRA sobre el frame con mezcla alfa."""
    if escala <= 0.02 or alfa <= 0.02:
        return
    h, w = png_bgra.shape[:2]
    nw, nh = max(2, int(w * escala)), max(2, int(h * escala))
    img = cv2.resize(png_bgra, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x0, y0 = int(cx - nw / 2), int(cy - nh / 2)
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(frame.shape[1], x0 + nw), min(frame.shape[0], y0 + nh)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    sub = img[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0].astype(np.float32)
    a = sub[:, :, 3:4] / 255.0 * alfa
    roi = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    frame[fy0:fy1, fx0:fx1] = (roi * (1 - a) + sub[:, :, :3] * a).astype(np.uint8)


def dibujar(frame, png, t_rel):
    """Anima el emoji: pop con rebote (0.28 s) → visible → fundido de salida."""
    DUR = 2.0
    if not 0 <= t_rel < DUR:
        return
    if t_rel < 0.28:
        esc = max(0.05, _ease_out_back(t_rel / 0.28))
        alfa = min(1.0, t_rel / 0.10)
    elif t_rel > DUR - 0.25:
        esc, alfa = 1.0, (DUR - t_rel) / 0.25
    else:
        esc, alfa = 1.0, 1.0
    pegar(frame, png, 540, 820, esc, alfa)
