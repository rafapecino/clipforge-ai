"""Análisis acústico del directo con librosa.

Extrae, en una rejilla temporal de 0.25 s:
- rms_db  : volumen (subidas de energía)
- arousal : "excitación" 0-1 (mezcla de energía + brillo espectral + onsets)
- gritos  : exceso de energía sobre el nivel habitual (picos fuertes)
"""
import numpy as np
from pathlib import Path

HOP_S = 0.25


def calcular(wav, cache=None, forzar=False):
    cache = Path(cache) if cache else None
    if cache and cache.exists() and not forzar:
        try:
            d = np.load(cache)
            feat = {k: d[k] for k in d.files}
            print(f"   💾 Rasgos de audio cargados de caché ({len(feat['t'])} ventanas)")
            return feat
        except Exception:
            pass
    import librosa
    y, sr = librosa.load(wav, sr=16000, mono=True)
    hop = int(sr * HOP_S)
    rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-9)
    cen = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    ons = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    n = min(len(rms_db), len(cen), len(ons))
    rms_db, cen, ons = rms_db[:n], cen[:n], ons[:n]
    t = np.arange(n) * HOP_S

    def z_robusta(x):
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-9
        return (x - med) / (1.4826 * mad)

    def suave(x, k=5):
        return np.convolve(x, np.ones(k) / k, mode="same")

    arousal = 0.55 * z_robusta(rms_db) + 0.25 * z_robusta(cen) + 0.20 * z_robusta(ons)
    arousal = np.clip(suave(arousal) * 0.32 + 0.5, 0, 1)
    gritos = np.clip(z_robusta(rms_db) - 1.8, 0, None)
    feat = {"t": t, "rms_db": rms_db, "arousal": arousal, "gritos": gritos}
    if cache:
        np.savez_compressed(cache, **feat)
    return feat


def stat(feat, campo, t0, t1, fn="mean"):
    """Estadístico de una serie en el rango temporal [t0, t1]."""
    i0, i1 = np.searchsorted(feat["t"], [t0, t1])
    seg = feat[campo][max(0, i0):max(i0 + 1, i1)]
    if len(seg) == 0:
        return 0.0
    return float(seg.max() if fn == "max" else seg.mean())


def picos(feat, min_sep=8.0, umbral=0.62):
    """Tiempos de picos de excitación separados al menos min_sep segundos."""
    a, t = feat["arousal"], feat["t"]
    out = []
    for i in np.argsort(a)[::-1]:
        if a[i] < umbral:
            break
        if all(abs(t[i] - x) >= min_sep for x in out):
            out.append(float(t[i]))
    return sorted(out)
