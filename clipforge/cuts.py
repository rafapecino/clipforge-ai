"""Eliminación de silencios y mapa temporal fuente→salida.

Construye una EDL (lista de rangos que se conservan) a partir de los
timestamps de palabra: las pausas más largas que `max_pausa` se comprimen.
TimeMap traduce tiempos en ambos sentidos (incluida la velocidad global)
para mantener sincronizados vídeo, audio, subtítulos, emojis y efectos.
"""
import numpy as np


def construir_edl(palabras, ini, fin, quitar=True, max_pausa=0.7,
                  pausa_restante=0.24, margen=0.12):
    ws = [w for w in palabras if w["s"] < fin and w["e"] > ini]
    if not quitar or not ws:
        return [(ini, fin)]
    rangos = []
    cur_s = max(ini, ws[0]["s"] - margen * 2)
    cur_e = ws[0]["e"]
    for w in ws[1:]:
        if w["s"] - cur_e > max_pausa:
            rangos.append((cur_s, min(fin, cur_e + pausa_restante / 2)))
            cur_s = max(ini, w["s"] - pausa_restante / 2)
        cur_e = w["e"]
    rangos.append((cur_s, min(fin, cur_e + margen * 3)))
    out = []
    for s, e in rangos:                      # fusionar contiguos, quitar migajas
        s, e = max(ini, s), min(fin, e)
        if out and s - out[-1][1] < 0.12:
            out[-1] = (out[-1][0], e)
        elif e - s > 0.15:
            out.append((s, e))
    return out or [(ini, fin)]


class TimeMap:
    """Mapa fuente↔salida sobre una EDL, con velocidad global opcional."""

    def __init__(self, edl, velocidad=1.0):
        self.edl = edl
        self.v = max(0.5, min(2.0, velocidad))
        src, out, acc = [], [], 0.0
        for s, e in edl:
            src += [s, e]
            out += [acc, acc + (e - s) / self.v]
            acc = out[-1]
        self.src_k = np.array(src)
        self.out_k = np.array(out)
        self.out_dur = acc

    def out2src(self, t):
        return float(np.interp(t, self.out_k, self.src_k))

    def src2out(self, t):
        # un tiempo dentro de un hueco eliminado cae en el corte
        return float(np.interp(t, self.src_k, self.out_k))

    def remap_palabras(self, palabras):
        out = []
        for w in palabras:
            for s, e in self.edl:
                if w["e"] > s and w["s"] < e:
                    a = self.src2out(max(w["s"], s))
                    b = self.src2out(min(w["e"], e))
                    if b - a > 0.03:
                        out.append({"w": w["w"], "s": round(a, 3), "e": round(b, 3)})
                    break
        return out
