"""Análisis semántico con sentence-transformers (multilingüe).

- viralidad: similitud de cada frase con "conceptos virales" (polémica,
  humor, dato sorprendente, exclusiva…)
- novedad  : distancia respecto al contexto anterior (cambios de tema)
"""
import numpy as np

MODELO = "paraphrase-multilingual-MiniLM-L12-v2"

CONSULTAS_VIRALES = [
    "una polémica o un escándalo muy fuerte",
    "una opinión contundente y sin filtros",
    "un dato sorprendente que nadie esperaba",
    "un momento muy gracioso y divertido",
    "una noticia importante de última hora",
    "una discusión o un enfrentamiento tenso",
    "una frase épica, lapidaria y memorable",
    "una crítica dura a alguien famoso",
    "dinero, cifras enormes y récords",
    "un anuncio o revelación exclusiva",
    "una caída o un accidente espectacular en una carrera de motos",
    "un adelantamiento increíble en la última vuelta",
    "una polémica sobre una sanción o la dirección de carrera",
    "un fichaje bomba o un rumor del mercado de pilotos",
]

_modelo = None


def _cargar():
    global _modelo
    if _modelo is None:
        from sentence_transformers import SentenceTransformer
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _modelo = SentenceTransformer(MODELO, device=dev)
    return _modelo


def calcular(textos):
    """Embeddings L2-normalizados de todas las frases."""
    m = _cargar()
    return m.encode(textos, batch_size=128, convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=True)


def viralidad(emb):
    """Máxima similitud de cada frase con los conceptos virales, 0-1."""
    q = _cargar().encode(CONSULTAS_VIRALES, convert_to_numpy=True,
                         normalize_embeddings=True)
    v = (emb @ q.T).max(axis=1)
    return (v - v.min()) / (v.max() - v.min() + 1e-9)


def novedad(emb, k=6):
    """1 - similitud con la media de las k frases previas (cambio de tema)."""
    n = len(emb)
    out = np.zeros(n)
    for i in range(1, n):
        ctx = emb[max(0, i - k):i].mean(axis=0)
        ctx /= (np.linalg.norm(ctx) + 1e-9)
        out[i] = 1.0 - float(emb[i] @ ctx)
    if out.max() > 0:
        out = np.clip(out / (np.percentile(out, 95) + 1e-9), 0, 1)
    return out
