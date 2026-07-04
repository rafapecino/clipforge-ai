"""Exportación a Drive: clip_NNN.mp4 + clip_NNN.json + RESUMEN.txt."""
import json
import shutil
from pathlib import Path

from .utils import t2str


def exportar_clip(ruta_local, job, dir_salida):
    dir_salida = Path(dir_salida)
    dir_salida.mkdir(parents=True, exist_ok=True)
    c = job["clip"]
    nombre = f"clip_{job['idx']:03d}"
    destino = dir_salida / f"{nombre}.mp4"
    shutil.copyfile(ruta_local, destino)
    meta = {
        "archivo": destino.name,
        "inicio": c["start"], "fin": c["end"],
        "inicio_hms": t2str(c["start"]), "fin_hms": t2str(c["end"]),
        "duracion_original": round(c["end"] - c["start"], 2),
        "duracion_clip": round(job["tm"].out_dur, 2),
        "score_viral": c.get("score", 0),
        "score_desglose": {"heuristico": c.get("score_heur", c.get("score")),
                           "llm": c.get("score_llm")},
        "tema": c.get("tema", ""),
        "resumen": c.get("resumen", ""),
        "keywords": c.get("keywords", []),
        "titulo": c.get("titulo", ""),
        "hook": c.get("hook", ""),
        "hashtags": c.get("hashtags", []),
        "emojis_texto": [m.get("emoji", "") for m in c.get("emojis", []) if isinstance(m, dict)],
        "n_emojis_en_video": len(job["emos"]),
        "caras_detectadas_pct": round(job["ruta"]["ratio"] * 100),
        "seleccion": c.get("fuente", ""),
    }
    (dir_salida / f"{nombre}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return destino, meta


def resumen_global(resultados, dir_salida, video):
    lineas = [f"CLIPS GENERADOS · {video}", "=" * 64, ""]
    for destino, meta in resultados:
        lineas += [f"▶ {meta['archivo']}  ·  {meta['inicio_hms']} → {meta['fin_hms']}"
                   f"  ·  ⭐ {meta['score_viral']}",
                   f"  Título:   {meta['titulo']}",
                   f"  Tema:     {meta['tema']}",
                   f"  Hashtags: {' '.join(meta['hashtags'])}", ""]
    p = Path(dir_salida) / "RESUMEN.txt"
    p.write_text("\n".join(lineas), encoding="utf-8")
    return p
