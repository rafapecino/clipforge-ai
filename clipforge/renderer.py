"""Render final de cada clip en 3 etapas:

1) Vídeo : lectura EDL del máster → reencuadre dinámico → efectos →
           emojis → barra de progreso → subtítulos ASS → x264 (tubería raw)
2) Audio : cortes EDL → mejora (highpass, de-noise, compresor, EQ,
           loudnorm -14 LUFS, limitador) → música opcional con ducking
3) Mux   : vídeo + audio + faststart
"""
import subprocess as sp
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from . import emojis as emo_mod
from .utils import run, hex_rgb

SALIDA_W, SALIDA_H = 1080, 1920


def _interp_ruta(ruta, t):
    return (float(np.interp(t, ruta["t"], ruta["cx"])),
            float(np.interp(t, ruta["t"], ruta["cy"])))


def render_video(src, job, cfg, tmp_video, fuente_dir="fonts"):
    tm, ruta, fx = job["tm"], job["ruta"], job["fx"]
    fps = cfg["fps"]
    n_frames = max(1, int(tm.out_dur * fps))
    W, H = job["W"], job["H"]
    base_cw = min(W, int(H * 9 / 16))
    pngs = {p: cv2.imread(p, cv2.IMREAD_UNCHANGED) for _, p in job["emos"]}
    color_barra = np.array(hex_rgb(cfg["color_resaltado"])[::-1], np.uint8)  # BGR
    vf = f"ass={Path(job['ass']).name}:fontsdir={fuente_dir}"
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{SALIDA_W}x{SALIDA_H}",
           "-r", str(fps), "-i", "pipe:",
           "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
           "-crf", str(cfg["crf"]), "-pix_fmt", "yuv420p", tmp_video]
    pr = sp.Popen(cmd, stdin=sp.PIPE, stderr=sp.PIPE, cwd=str(Path(job["ass"]).parent))
    cap = cv2.VideoCapture(src)
    cur_t, frame_src, prev_out = -1.0, None, None
    try:
        for n in tqdm(range(n_frames), desc=f"   🎬 clip_{job['idx']:03d}",
                      unit="fr", leave=False):
            t_out = n / fps
            t_src = tm.out2src(t_out)
            if t_src < cur_t - 0.05 or t_src > cur_t + 1.5:     # salto de rango EDL
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_src * 1000.0 - 40))
                cur_t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                frame_src = None
            intentos = 0
            while (frame_src is None or cur_t < t_src - 0.001) and intentos < 90:
                pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                ok, fr = cap.read()
                if not ok:
                    break
                frame_src, cur_t = fr, pos
                intentos += 1
            if frame_src is None:
                frame_out = np.zeros((SALIDA_H, SALIDA_W, 3), np.uint8)
            else:
                z = fx.zoom(t_out)
                dx, dy = fx.shake(t_out)
                cw, ch = base_cw / z, H / z
                cx, cy = _interp_ruta(ruta, t_out)
                x0 = int(np.clip(cx - cw / 2 + dx, 0, W - cw))
                y0 = int(np.clip(cy - ch / 2 + dy, 0, H - ch))
                rec = frame_src[y0:y0 + int(ch), x0:x0 + int(cw)]
                interp = cv2.INTER_AREA if cw > SALIDA_W else cv2.INTER_LINEAR
                frame_out = cv2.resize(rec, (SALIDA_W, SALIDA_H), interpolation=interp)
                if cfg.get("motion_blur") and prev_out is not None and fx.dzoom(t_out) > 0.9:
                    frame_out = cv2.addWeighted(frame_out, 0.65, prev_out, 0.35, 0)
                fa = fx.flash(t_out)
                if fa > 0:
                    frame_out = cv2.addWeighted(frame_out, 1 - fa,
                                                np.full_like(frame_out, 255), fa, 0)
            prev_out = frame_out.copy() if cfg.get("motion_blur") else None
            for t0, p in job["emos"]:
                png = pngs.get(p)
                if png is not None:
                    emo_mod.dibujar(frame_out, png, t_out - t0)
            if cfg.get("barra"):
                ancho = int(SALIDA_W * min(1.0, t_out / max(tm.out_dur, 0.1)))
                frame_out[SALIDA_H - 12:, :ancho] = color_barra
            try:
                pr.stdin.write(frame_out.tobytes())
            except (BrokenPipeError, OSError):
                break
    finally:
        cap.release()
        if pr.stdin:
            pr.stdin.close()
        err = pr.stderr.read().decode(errors="ignore") if pr.stderr else ""
        if pr.wait() != 0:
            raise RuntimeError("ffmpeg (vídeo) falló:\n" + err[-1200:])
    return tmp_video


def render_audio(src, job, cfg, tmp_audio, ruta_musica=None):
    ini, fin = job["clip"]["start"], job["clip"]["end"]
    tm = job["tm"]
    wav_src = tmp_audio.replace(".wav", "_src.wav")
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{ini:.3f}",
         "-t", f"{fin - ini:.3f}", "-i", src, "-vn", "-ac", "2", "-ar", "48000", wav_src])
    n = len(tm.edl)
    if n == 1:
        s, e = tm.edl[0]
        fc = f"[0:a]atrim={s - ini:.3f}:{e - ini:.3f},asetpts=PTS-STARTPTS[cat];"
    else:
        splits = "".join(f"[i{i}]" for i in range(n))
        partes = [f"[0:a]asplit={n}{splits}"]
        for i, (s, e) in enumerate(tm.edl):
            partes.append(f"[i{i}]atrim={s - ini:.3f}:{e - ini:.3f},asetpts=PTS-STARTPTS[s{i}]")
        fc = (";".join(partes) + ";" + "".join(f"[s{i}]" for i in range(n)) +
              f"concat=n={n}:v=0:a=1[cat];")
    tempo = f"atempo={tm.v:.3f}," if abs(tm.v - 1) > 1e-3 else ""
    fc += ("[cat]highpass=f=70,afftdn=nf=-25,"
           "acompressor=threshold=-18dB:ratio=3:attack=6:release=180,"
           "equalizer=f=3000:t=q:w=1:g=1.5," + tempo +
           "loudnorm=I=-14:TP=-1.5:LRA=11,alimiter=limit=0.97,aresample=48000[voz]")
    entradas = ["-i", wav_src]
    mapa = "[voz]"
    if ruta_musica:
        d = tm.out_dur
        entradas += ["-stream_loop", "-1", "-i", ruta_musica]
        fc += (f";[voz]asplit[v1][v2];"
               f"[1:a]atrim=0:{d:.2f},asetpts=PTS-STARTPTS,volume={cfg['vol_musica']:.2f},"
               f"afade=t=in:d=1.2,afade=t=out:st={max(0, d - 2):.2f}:d=2[mus];"
               f"[mus][v1]sidechaincompress=threshold=0.02:ratio=12:attack=5:release=400[duck];"
               f"[v2][duck]amix=inputs=2:duration=first:weights=3 1[mezcla]")
        mapa = "[mezcla]"
    run(["ffmpeg", "-y", "-loglevel", "error"] + entradas +
        ["-filter_complex", fc, "-map", mapa, "-ar", "48000", "-ac", "2", tmp_audio])
    return tmp_audio


def render_clip(src, job, cfg, dir_tmp, fuente_dir="fonts", ruta_musica=None):
    dir_tmp = Path(dir_tmp)
    base = f"clip_{job['idx']:03d}"
    tv = str(dir_tmp / f"{base}_v.mp4")
    ta = str(dir_tmp / f"{base}_a.wav")
    out = str(dir_tmp / f"{base}.mp4")
    render_video(src, job, cfg, tv, fuente_dir)
    render_audio(src, job, cfg, ta, ruta_musica)
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", tv, "-i", ta,
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
         "-movflags", "+faststart", out])
    return out
