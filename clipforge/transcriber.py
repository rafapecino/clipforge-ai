"""Transcripción con faster-whisper (CTranslate2) + timestamps por palabra.

Usa el VAD de Silero integrado para saltar zonas sin voz y evitar
alucinaciones en música/silencios. Cachea el resultado en Drive con una
clave (tamaño del vídeo + modelo) para no recalcular en re-ejecuciones.
"""
import json
import re
from pathlib import Path


def _preparar_cuda():
    """Pre-carga cuDNN/cuBLAS instaladas por pip para que CTranslate2 las
    encuentre (fallo clásico de faster-whisper en Colab)."""
    import ctypes
    import glob
    for pat in ("/usr/local/lib/python3*/dist-packages/nvidia/cublas/lib/*.so*",
                "/usr/local/lib/python3*/dist-packages/nvidia/cudnn/lib/*.so*"):
        for so in sorted(glob.glob(pat)):
            try:
                ctypes.CDLL(so)
            except OSError:
                pass


def transcribir(wav, modelo="large-v3", idioma="es", cache=None, forzar=False,
                clave_video=None, progreso=None):
    """Devuelve (palabras, info) con palabras = [{'w','s','e','p'}, ...]."""
    cache = Path(cache) if cache else None
    if cache and cache.exists() and not forzar:
        try:
            d = json.loads(cache.read_text(encoding="utf-8"))
            if d.get("clave") == clave_video and d.get("modelo") == modelo:
                print(f"   💾 Transcripción cargada de caché ({len(d['palabras'])} palabras)")
                return d["palabras"], d["info"]
        except Exception:
            pass
    _preparar_cuda()
    from faster_whisper import WhisperModel
    wm = None
    for dev, ct in (("cuda", "float16"), ("cuda", "int8_float16"), ("cpu", "int8")):
        try:
            print(f"   Cargando Whisper {modelo} en {dev} ({ct})…")
            wm = WhisperModel(modelo, device=dev, compute_type=ct)
            break
        except Exception as e:
            print(f"   ⚠️ {dev}/{ct}: {str(e)[:120]}")
    if wm is None:
        raise RuntimeError("No se pudo cargar Whisper en ningún dispositivo")
    segs, inf = wm.transcribe(wav, language=idioma or None, vad_filter=True,
                              word_timestamps=True, beam_size=5)
    palabras = []
    for seg in segs:
        for w in (seg.words or []):
            t = w.word.strip()
            if t:
                palabras.append({"w": t, "s": round(w.start, 3),
                                 "e": round(w.end, 3), "p": round(w.probability, 3)})
        if progreso:
            progreso(min(seg.end, inf.duration), inf.duration)
    info = {"dur": inf.duration, "idioma": inf.language, "n": len(palabras)}
    if cache:
        cache.write_text(json.dumps({"clave": clave_video, "modelo": modelo,
                                     "info": info, "palabras": palabras},
                                    ensure_ascii=False), encoding="utf-8")
    return palabras, info


_FIN = re.compile(r"[\.\?\!…]$")


def construir_frases(palabras, gap_corte=0.8, max_palabras=42):
    """Agrupa palabras en frases usando puntuación y pausas."""
    frases, cur = [], []
    for i, w in enumerate(palabras):
        cur.append(w)
        gap = palabras[i + 1]["s"] - w["e"] if i + 1 < len(palabras) else 99.0
        if _FIN.search(w["w"]) or gap > gap_corte or len(cur) >= max_palabras:
            frases.append(_cerrar(cur, len(frases)))
            cur = []
    if cur:
        frases.append(_cerrar(cur, len(frases)))
    return frases


def _cerrar(ws, i):
    return {"i": i, "texto": " ".join(x["w"] for x in ws),
            "s": ws[0]["s"], "e": ws[-1]["e"], "n": len(ws)}
