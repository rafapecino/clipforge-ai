"""Utilidades comunes: subprocesos, ffprobe, tiempos y colores."""
import json
import subprocess
import time


def run(cmd, **kw):
    """Ejecuta un comando y lanza un error legible si falla."""
    cmd = [str(c) for c in cmd]
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError("Comando fallido: " + " ".join(cmd)[:300] +
                           "\n--- stderr ---\n" + (r.stderr or "")[-1500:])
    return r


def ffprobe_info(path):
    """Resolución, fps, duración y tamaño de un vídeo."""
    r = run(["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", path])
    d = json.loads(r.stdout)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    try:
        num, den = (v.get("avg_frame_rate") or "30/1").split("/")
        fps = float(num) / float(den) if float(den) else 30.0
    except Exception:
        fps = 30.0
    if not 5 < fps < 121:
        fps = 30.0
    return {"w": int(v["width"]), "h": int(v["height"]), "fps": fps,
            "dur": float(d["format"]["duration"]), "bytes": int(d["format"]["size"])}


def t2str(t):
    """Segundos → 'H:MM:SS' o 'MM:SS'."""
    t = max(0, int(round(t)))
    h, m, s = t // 3600, t % 3600 // 60, t % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class Paso:
    """Context manager que cronometra y muestra el estado de cada paso."""

    def __init__(self, nombre):
        self.nombre = nombre

    def __enter__(self):
        print(f"▶ {self.nombre}…", flush=True)
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, *a):
        icono = "✅" if exc_type is None else "❌"
        print(f"{icono} {self.nombre} · {time.time() - self.t0:.1f}s", flush=True)


def hex_rgb(hx):
    hx = hx.lstrip("#")
    return int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)


def hex_ass(hx):
    """'#RRGGBB' → color ASS '&H00BBGGRR'."""
    r, g, b = hex_rgb(hx)
    return f"&H00{b:02X}{g:02X}{r:02X}"
