"""Puntuación multi-señal de frases y generación de clips candidatos.

Cada frase recibe un score 0-10 que combina:
  kw     · palabras clave virales en español      punct · ¡! y ¿?
  audio  · excitación acústica (energía, gritos)  ritmo · velocidad al hablar
  sem    · similitud semántica con conceptos virales (embeddings)
  nov    · novedad / cambio de tema

Los candidatos son ventanas de frases ajustadas a duraciones objetivo
(20/30/45/60 s) y se filtran por solapamiento.
"""
import math
import re
import numpy as np
from . import audio_features as af

PESOS = {"kw": 1.0, "audio": 1.3, "sem": 1.1, "nov": 0.6, "punct": 0.5, "ritmo": 0.4}

PALABRAS_CLAVE = {
    # impacto / sorpresa
    "increíble": 3, "increible": 3, "brutal": 3, "locura": 3, "tremendo": 3,
    "bestial": 3, "alucinante": 3, "flipante": 3, "flipa": 3, "flipando": 3,
    "espectacular": 2, "impresionante": 2, "no puede ser": 4, "no me lo creo": 3,
    "madre mía": 3, "ojo": 2, "atención": 2, "escucha": 2, "bomba": 4, "bombazo": 4,
    # polémica / conflicto
    "polémica": 4, "polémico": 3, "escándalo": 4, "vergüenza": 3, "vergonzoso": 3,
    "desastre": 3, "ridículo": 3, "culpa": 2, "mentira": 3, "engaño": 3,
    "guerra": 2, "ataque": 2, "golpe": 3, "hostia": 3, "joder": 2, "cabreo": 3,
    # datos / dinero
    "récord": 3, "histórico": 3, "millones": 3, "dinero": 3, "euros": 2,
    "secreto": 3, "exclusiva": 3, "confirmado": 3, "oficial": 3, "última hora": 4,
    # narrativa
    "nunca": 2, "jamás": 2, "nadie": 2, "mejor": 2, "peor": 2,
    "verdad": 2, "importante": 2, "clave": 2, "gravísimo": 3, "grave": 2,
    # motor / MotoGP (contexto del canal)
    "adelantamiento": 2, "accidente": 3, "caída": 3, "sanción": 3, "multa": 2,
    "descalificado": 3, "pole": 2, "victoria": 2, "campeón": 2, "mundial": 2, "podio": 2,
    "márquez": 2, "marquez": 2, "bagnaia": 2, "pecco": 2, "acosta": 2,
    "quartararo": 2, "martín": 2, "viñales": 2, "rossi": 2, "lorenzo": 2,
    "ducati": 2, "yamaha": 2, "honda": 2, "ktm": 2, "aprilia": 2, "motogp": 2,
}


def _kw(texto):
    t = " " + texto.lower() + " "
    return sum(p for k, p in PALABRAS_CLAVE.items() if k in t)


def puntuar_frases(frases, feat, viral, nov):
    """Añade a cada frase su 'score' 0-10 y el desglose por señal."""
    for f in frases:
        dur = max(f["e"] - f["s"], 0.4)
        c = {
            "kw": min(_kw(f["texto"]) / 6.0, 1.0),
            "audio": min(af.stat(feat, "arousal", f["s"], f["e"]) +
                         0.5 * af.stat(feat, "gritos", f["s"], f["e"], "max"), 1.2),
            "sem": float(viral[f["i"]]),
            "nov": float(nov[f["i"]]),
            "punct": min((f["texto"].count("!") + f["texto"].count("?")) / 2.0, 1.0),
            "ritmo": 1 / (1 + math.exp(-(f["n"] / dur - 3.2))),
        }
        f["score"] = round(sum(PESOS[k] * v for k, v in c.items()) / sum(PESOS.values()) * 10, 2)
        f["senales"] = {k: round(v, 3) for k, v in c.items()}
    return frases


_TERMINAL = re.compile(r"[\.\?\!…]$")


def candidatos(frases, num, duraciones=(20, 30, 45, 60), dur_min=18, dur_max=68, umbral=2.0):
    """Genera ventanas candidatas y devuelve las mejores sin solaparse.

    Para cada frase de arranque prueba las 4 duraciones objetivo y se queda
    con la mejor: así el sistema decide la duración según el contenido.
    """
    cands = []
    N = len(frases)
    for i in range(N):
        if frases[i]["score"] < umbral:   # el gancho inicial debe ser potente
            continue
        mejor = None
        for objetivo in duraciones:
            k = i
            while k < N and frases[k]["e"] - frases[i]["s"] <= dur_max:
                dur = frases[k]["e"] - frases[i]["s"]
                if dur >= dur_min and abs(dur - objetivo) <= objetivo * 0.25:
                    span = frases[i:k + 1]
                    media = sum(f["score"] for f in span) / len(span)
                    maxi = max(f["score"] for f in span)
                    gancho = sum(f["score"] for f in span[:2]) / min(2, len(span))
                    sc = 0.5 * media + 0.25 * maxi + 0.25 * gancho
                    if _TERMINAL.search(frases[k]["texto"]):
                        sc += 0.35                      # cierre natural
                    sc -= abs(dur - objetivo) / objetivo * 0.4
                    if mejor is None or sc > mejor[0]:
                        mejor = (sc, i, k, dur, objetivo)
                k += 1
        if mejor:
            cands.append(mejor)
    cands.sort(key=lambda x: -x[0])
    eleg = []
    for sc, i, k, dur, obj in cands:
        s, e = frases[i]["s"], frases[k]["e"]
        if any(min(e, x["end"]) - max(s, x["start"]) > 0.2 * dur for x in eleg):
            continue
        eleg.append({"start": round(s, 2), "end": round(e, 2), "dur": round(dur, 1),
                     "objetivo": obj, "i0": i, "i1": k,
                     "score": int(np.clip(sc * 10 + 12, 5, 99))})
        if len(eleg) >= num:
            break
    eleg.sort(key=lambda c: c["start"])
    return eleg
