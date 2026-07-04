"""Subtítulos ASS estilo Opus Clip / CapCut / Submagic:

- páginas de máx. 3 palabras (nunca llenan la pantalla)
- karaoke palabra a palabra: la palabra activa se resalta en color con
  rebote (escala 86 → 118 → 100) y el resto queda en blanco
- blanco + borde negro grueso + sombra, tamaño adaptativo
- entrada/salida suaves por página, rótulo superior (hook) y marca de agua
"""
import re

_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2300-\u23FF\u2B00-\u2BFF\u2190-\u21FF\u2122\u00A9\u00AE\uFE0F\u200D]")


def _limpiar(t):
    t = _EMOJI.sub("", t).replace("{", "(").replace("}", ")").replace("\\", "/")
    return re.sub(r"\s+", " ", t).strip()


def _t(t):
    t = max(0.0, t)
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    return f"{h}:{m:02d}:{t % 60:05.2f}"


def _hex_ass(hx):
    hx = hx.lstrip("#")
    return "&H00" + hx[4:6].upper() + hx[2:4].upper() + hx[0:2].upper()


def paginar(palabras, max_pal=3, max_dur=1.6, corte_gap=0.6):
    pags, cur = [], []
    for i, w in enumerate(palabras):
        cur.append(w)
        gap = palabras[i + 1]["s"] - w["e"] if i + 1 < len(palabras) else 9.0
        puntua = re.search(r"[\.\?\!…,;:]$", w["w"]) is not None
        if (len(cur) >= max_pal or cur[-1]["e"] - cur[0]["s"] > max_dur
                or gap > corte_gap or (puntua and len(cur) >= 2)):
            pags.append(cur)
            cur = []
    if cur:
        pags.append(cur)
    return pags


def generar_ass(palabras, ruta, dur, fuente="Poppins ExtraBold", color="#FFE600",
                mayusculas=True, hook="", marca="", con_subs=True):
    HL = _hex_ass(color)
    L = ["[Script Info]", "ScriptType: v4.00+", "PlayResX: 1080", "PlayResY: 1920",
         "WrapStyle: 0", "ScaledBorderAndShadow: yes", "", "[V4+ Styles]",
         ("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
          "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
          "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
          "MarginL, MarginR, MarginV, Encoding"),
         f"Style: Cap,{fuente},92,&H00FFFFFF,&H00FFFFFF,&H00101010,&H9B000000,0,0,0,0,100,100,0,0,1,8,3,2,60,60,600,1",
         f"Style: Head,{fuente},54,&H00FFFFFF,&H00FFFFFF,&H87000000,&H87000000,0,0,0,0,100,100,0,0,3,7,0,8,80,80,116,1",
         f"Style: WM,{fuente},34,&H7DFFFFFF,&H00FFFFFF,&H7D101010,&H00000000,0,0,0,0,100,100,0,0,1,2,0,3,40,36,52,1",
         "", "[Events]",
         "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    if hook:
        L.append(f"Dialogue: 2,{_t(0)},{_t(dur)},Head,,0,0,0,," +
                 "{\\fad(220,220)}" + _limpiar(hook).upper())
    if marca:
        L.append(f"Dialogue: 2,{_t(0)},{_t(dur)},WM,,0,0,0,," + _limpiar(marca))
    if con_subs:
        for pag in paginar(palabras):
            fin_pag = pag[-1]["e"]
            nch = sum(len(w["w"]) for w in pag) + len(pag) - 1
            fs = "{\\fs64}" if nch > 24 else ("{\\fs76}" if nch > 17 else "")
            for j, w in enumerate(pag):
                a = w["s"]
                b = pag[j + 1]["s"] if j + 1 < len(pag) else fin_pag
                if b - a < 0.06:
                    b = a + 0.06
                partes = []
                for k, x in enumerate(pag):
                    tx = _limpiar(re.sub(r"[\.,;:]+$", "", x["w"]))
                    if mayusculas:
                        tx = tx.upper()
                    if not tx:
                        continue
                    if k == j:   # palabra activa: color + rebote
                        partes.append("{\\c" + HL + "\\fscx86\\fscy86"
                                      "\\t(0,70,\\fscx118\\fscy118)"
                                      "\\t(70,150,\\fscx100\\fscy100)}" + tx + "{\\r}")
                    else:
                        partes.append(tx)
                extra = "{\\fad(90,0)}" if j == 0 else ("{\\fad(0,70)}" if j == len(pag) - 1 else "")
                L.append(f"Dialogue: 0,{_t(a)},{_t(b)},Cap,,0,0,0,,{extra}{fs}" + " ".join(partes))
    ruta.write_text("\n".join(L) + "\n", encoding="utf-8")
    return ruta
