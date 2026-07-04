"""Director editorial con LLM (Gemini) + respaldo heurístico completo.

El LLM recibe los candidatos con su transcripción, elige los mejores,
puede ajustar los límites (±4 frases) y genera título, hook, tema,
resumen, keywords, hashtags, emojis con cita exacta y score 0-100.
Si no hay API key o el servicio falla, todo se genera heurísticamente:
el pipeline nunca se detiene.
"""
import json
import os
import re
from collections import Counter

MODELOS = ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash")

STOP = set(("de la que el en y a los se del las un por con no una su para es al lo como "
            "más pero sus le ya o fue este ha sí porque esta son entre cuando muy sin sobre "
            "ser tiene también me hasta hay donde quien desde todo nos durante todos uno les "
            "ni contra otros ese eso ante ellos e esto mí antes algunos qué unos yo otro otras "
            "otra él tanto esa estos mucho quienes nada muchos cual poco ella estar estas "
            "algunas algo nosotros vamos estamos tiene tenemos porque entonces bueno pues").split())

PLANTILLA = """Eres el director editorial de clips virales de un canal de YouTube.
Contexto del canal: <<CTX>>

Te paso <<NC>> momentos candidatos de un directo, con sus frases transcritas.
Elige los <<N>> MEJORES para TikTok/Shorts/Reels y devuelve SOLO JSON válido.

Para cada clip elegido:
- "id": id del candidato.
- "usar": true.
- "ajuste_ini" / "ajuste_fin": desplazamiento en nº de frases (-4 a 4) si mejora el gancho o el cierre.
- "titulo": español, viral, máx 70 caracteres, 1-2 emojis, sin comillas.
- "hook": rótulo corto para sobreimprimir en el vídeo, máx 45 caracteres, SIN emojis.
- "tema": 2-4 palabras.
- "resumen": máx 180 caracteres.
- "keywords": 5 palabras clave.
- "hashtags": 5 hashtags en español.
- "score": viralidad 0-100 (sé exigente, no regales puntos).
- "emojis": 2-5 objetos {"quote": "2-4 palabras EXACTAS de la transcripción", "emoji": "🔥"}.

Criterios: gancho en los 2 primeros segundos, polémica, opiniones fuertes,
datos sorprendentes, humor, tensión, cierre natural. Descarta lo aburrido.

Formato de respuesta:
{"clips": [{"id": 3, "usar": true, "ajuste_ini": 0, "ajuste_fin": 1, "titulo": "…", "hook": "…", "tema": "…", "resumen": "…", "keywords": ["…"], "hashtags": ["#…"], "score": 87, "emojis": [{"quote": "…", "emoji": "🔥"}]}]}

CANDIDATOS:
<<CANDS>>"""


def clave_api():
    try:
        from google.colab import userdata
        k = userdata.get("GEMINI_API_KEY")
        if k:
            return k
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")


def _texto_candidatos(cands, frases):
    bloques = []
    for n, c in enumerate(cands):
        frs = frases[c["i0"]:c["i1"] + 1]
        txt = " ".join(f["texto"] for f in frs)[:900]
        m0, s0 = divmod(int(c["start"]), 60)
        bloques.append(f"[id {n} · min {m0:02d}:{s0:02d} · {c['dur']:.0f}s · score base {c['score']}]\n{txt}")
    return "\n\n".join(bloques)


def _extraer_json(texto):
    texto = re.sub(r"^```(json)?|```$", "", texto.strip(), flags=re.M).strip()
    m = re.search(r"\{[\s\S]*\}", texto)
    return json.loads(m.group(0) if m else texto)


def dirigir(cands, frases, num, contexto="", canal="PecinoGP"):
    """Devuelve el plan final de clips con metadatos completos."""
    plan = None
    key = clave_api()
    if key:
        plan = _con_gemini(cands, frases, num, contexto, key)
    if plan is None:
        print("   ℹ️ Selección y títulos heurísticos. Añade GEMINI_API_KEY en los "
              "Secretos 🔑 de Colab (gratis: aistudio.google.com/app/apikey) para mejorarlos.")
        plan = []
    for c in sorted(cands, key=lambda x: -x["score"]):
        if len(plan) >= num:
            break
        if any(min(c["end"], p["end"]) - max(c["start"], p["start"]) > 3 for p in plan):
            continue
        plan.append(_meta_heuristica(c, frases, canal))
    plan = sorted(plan[:num], key=lambda c: c["start"])
    for j, c in enumerate(plan, 1):
        c["id"] = j
    return plan


def _con_gemini(cands, frases, num, contexto, key):
    prompt = (PLANTILLA.replace("<<CTX>>", contexto).replace("<<N>>", str(num))
              .replace("<<NC>>", str(len(cands)))
              .replace("<<CANDS>>", _texto_candidatos(cands, frases)))
    try:
        from google import genai
        cli = genai.Client(api_key=key)
    except Exception as e:
        print(f"   ⚠️ google-genai no disponible: {e}")
        return None
    for modelo in MODELOS:
        try:
            print(f"   🤖 Analizando candidatos con {modelo}…")
            r = cli.models.generate_content(
                model=modelo, contents=prompt,
                config={"response_mime_type": "application/json", "temperature": 0.4})
            data = _extraer_json(r.text or "")
            plan = []
            for c in data.get("clips", []):
                if not c.get("usar", True):
                    continue
                base = cands[int(c["id"])]
                i0 = max(0, min(len(frases) - 1, base["i0"] + int(c.get("ajuste_ini", 0) or 0)))
                i1 = max(i0, min(len(frases) - 1, base["i1"] + int(c.get("ajuste_fin", 0) or 0)))
                s, e = frases[i0]["s"], frases[i1]["e"]
                if not 12 <= e - s <= 90:
                    i0, i1, s, e = base["i0"], base["i1"], base["start"], base["end"]
                plan.append({
                    "start": round(s, 2), "end": round(e, 2), "dur": round(e - s, 1),
                    "i0": i0, "i1": i1,
                    "score": int(round(0.55 * int(c.get("score", base["score"])) + 0.45 * base["score"])),
                    "score_llm": int(c.get("score", 0)), "score_heur": base["score"],
                    "titulo": str(c.get("titulo", "")).strip()[:80] or "🔥 Momentazo del directo",
                    "hook": str(c.get("hook", "")).strip()[:48],
                    "tema": str(c.get("tema", "directo"))[:40],
                    "resumen": str(c.get("resumen", ""))[:220],
                    "keywords": [str(x) for x in c.get("keywords", [])][:6],
                    "hashtags": [str(x) for x in c.get("hashtags", [])][:6],
                    "emojis": [x for x in c.get("emojis", []) if isinstance(x, dict)][:5],
                    "fuente": modelo,
                })
            if plan:
                plan.sort(key=lambda p: p["start"])
                limpio = []
                for p in plan:
                    if limpio and p["start"] < limpio[-1]["end"] - 2:
                        if p["score"] > limpio[-1]["score"]:
                            limpio[-1] = p
                        continue
                    limpio.append(p)
                return limpio[:num]
        except Exception as e:
            print(f"   ⚠️ {modelo}: {type(e).__name__} {str(e)[:150]}")
    return None


def _meta_heuristica(c, frases, canal):
    span = frases[c["i0"]:c["i1"] + 1]
    top = max(span, key=lambda f: f["score"])
    titulo = top["texto"].strip().rstrip(".")
    titulo = (titulo[:62] + "…") if len(titulo) > 64 else titulo
    tokens = [t for t in re.findall(r"[a-záéíóúñü]{4,}",
                                    " ".join(f["texto"] for f in span).lower())
              if t not in STOP]
    kws = [w for w, _ in Counter(tokens).most_common(5)]
    out = dict(c)
    out.update({
        "titulo": "🔥 " + titulo.capitalize(),
        "hook": titulo[:45].upper(),
        "tema": kws[0] if kws else "directo",
        "resumen": " ".join(f["texto"] for f in span)[:200],
        "keywords": kws,
        "hashtags": ["#shorts", "#viral", "#motogp", f"#{canal.lower()}", "#directo"],
        "emojis": [],
        "fuente": "heuristica",
    })
    return out
