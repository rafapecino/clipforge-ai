# 🎬 ClipForge AI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rafapecino/clipforge-ai/blob/main/ClipForge_AI_PecinoGP.ipynb)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Made for](https://img.shields.io/badge/canal-PecinoGP%20🏍️-red)

**Tu Opus Clip privado, gratis y sin límites** — un pipeline de IA que convierte directos completos de YouTube en clips verticales 9:16 listos para TikTok / Shorts / Reels, ejecutándose en Google Colab sobre vídeos de Google Drive.

> Pulsa el badge **Open in Colab** de arriba, selecciona GPU T4 y dale a *Ejecutar todas*. Eso es todo.

---

## ✨ Qué hace

| Función | Cómo lo hace |
|---|---|
| 🧠 Detección de momentos virales | Whisper + análisis acústico (librosa) + embeddings semánticos + LLM director (Gemini) |
| ⭐ Score de viralidad 0-100 | Fusión de 6 señales por frase + veredicto del LLM |
| ✂️ Duración automática | Elige entre 20 / 30 / 45 / 60 s según el contenido |
| 🗣️ Subtítulos profesionales | Karaoke palabra a palabra con rebote, blanco + borde negro + resaltado amarillo, máx. 3 palabras |
| 😂 Emojis contextuales | Solo cuando tienen sentido (citas del LLM o mapa semántico), con animación *pop* |
| 🎥 Reencuadre inteligente 16:9 → 9:16 | MediaPipe sigue la cara del hablante con paneo suave e histéresis anti-saltos |
| 🔇 Eliminación de silencios | Comprime pausas largas manteniendo la sincronía total (EDL + TimeMap) |
| ⚡ Efectos dinámicos | Zoom in/out, punch zoom en picos, shake, flash, motion blur |
| 🎚️ Audio broadcast | De-noise, compresor, EQ, loudnorm −14 LUFS, limitador, música opcional con ducking |
| 📦 Export + metadatos | `clip_001.mp4` + `clip_001.json` (score, tema, resumen, keywords, hashtags) + `RESUMEN.txt` |

## 🏗️ Arquitectura

```text
VÍDEO (Drive) ──► COPIA LOCAL ──► ① WHISPER large-v3 · palabra a palabra · VAD Silero
                                  ② AUDIO (librosa): energía, brillo, gritos, picos
                                  ③ SEMÁNTICA: embeddings multilingües + novedad de tema
                                                │
                               ④ FUSIÓN DE SEÑALES por frase (score 0-10)
                                                │
                               ⑤ CANDIDATOS 20/30/45/60 s + anti-solapamiento
                                                │
                               ⑥ DIRECTOR LLM (Gemini): elige, ajusta límites,
                                  títulos, hooks, temas, emojis, score 0-100
                                                │
        ┌──────────────┬──────────────────┬─────┴──────────┬───────────────┐
   ⑦ SILENCIOS     ⑧ REENCUADRE      ⑨ SUBTÍTULOS      ⑩ EFECTOS       ⑪ AUDIO PRO
        └──────────────┴──────────────────┴────────────────┴───────────────┘
                                                │
                     ⑫ RENDER (OpenCV frame a frame → tubería ffmpeg)
                            └──► Clips/clip_NNN.mp4 + clip_NNN.json + RESUMEN.txt
```

El código vive en el paquete modular [`clipforge/`](clipforge) (15 módulos pequeños y sustituibles). El notebook es **autocontenido**: escribe su propia copia del paquete en Colab mediante celdas `%%writefile`; la carpeta `clipforge/` de este repo contiene el mismo código para leerlo cómodamente o reutilizarlo fuera de Colab.

| Módulo | Responsabilidad |
|---|---|
| [`transcriber.py`](clipforge/transcriber.py) | faster-whisper palabra a palabra + frases + caché |
| [`audio_features.py`](clipforge/audio_features.py) | energía, brillo, gritos, picos de excitación |
| [`semantics.py`](clipforge/semantics.py) | embeddings, similitud con conceptos virales, novedad |
| [`scoring.py`](clipforge/scoring.py) | fusión de señales + ventanas candidatas |
| [`llm_director.py`](clipforge/llm_director.py) | Gemini (títulos/emojis/score) + respaldo heurístico |
| [`cuts.py`](clipforge/cuts.py) | eliminación de silencios (EDL) + mapa temporal |
| [`reframe.py`](clipforge/reframe.py) | seguimiento facial y trayectoria de encuadre |
| [`subtitles.py`](clipforge/subtitles.py) | ASS karaoke animado estilo Opus Clip |
| [`emojis.py`](clipforge/emojis.py) | PNG Noto Emoji + animación pop |
| [`effects.py`](clipforge/effects.py) | zoom / punch / shake / flash / motion blur |
| [`renderer.py`](clipforge/renderer.py) | render frame a frame + audio pro + mux |
| [`exporter.py`](clipforge/exporter.py) | clips numerados + JSON de metadatos + resumen |

## 🚀 Uso

1. **[Abre el notebook en Colab](https://colab.research.google.com/github/rafapecino/clipforge-ai/blob/main/ClipForge_AI_PecinoGP.ipynb)** y selecciona `Entorno de ejecución ▸ Cambiar tipo de entorno de ejecución ▸ T4 GPU`.
2. *(Recomendado, gratis)* Crea una API key en [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) y guárdala en los **Secretos 🔑 de Colab** como `GEMINI_API_KEY` (activa "Acceso desde el cuaderno"). Sin ella, el pipeline funciona en modo heurístico.
3. Ajusta en la sección 2 la carpeta y el nombre de tu vídeo de Drive (`MyDrive/VideosGP/…`) y pulsa `Ejecutar todas`.

Los clips aparecen en `MyDrive/<CARPETA_SALIDA>/` con sus metadatos. Todo se cachea en Drive: re-ejecutar tarda segundos y puedes editar `plan_clips.json` a mano para ajustar momentos o títulos y re-renderizar.

### ⚙️ Configuración principal (sección 2 del notebook)

| Parámetro | Efecto |
|---|---|
| `NUM_CLIPS`, `DURACION_MINIMA/MAXIMA` | cuántos clips y entre qué duraciones |
| `TAMANO_WHISPER` | `large-v3` (máxima calidad) o `large-v3-turbo` (~4× más rápido) |
| `SUBTITULOS / EMOJIS / EFECTOS_ZOOM / RECORTE_INTELIGENTE / QUITAR_SILENCIOS / MUSICA` | activar o desactivar cada función |
| `FUENTE`, `COLOR_RESALTADO`, `MARCA_DE_AGUA`, `BARRA_PROGRESO` | estilo visual |
| `VELOCIDAD_GLOBAL` | acelera el clip (1.0–1.15×) manteniendo sincronizados subtítulos y efectos |

### ⏱️ Rendimiento orientativo (directo de 2 h, T4 gratuita)

| Etapa | Tiempo |
|---|---|
| Transcripción | ≈ 25-40 min (`large-v3`) · ≈ 10-12 min (`large-v3-turbo`) |
| Análisis IA (audio + embeddings + selección) | ≈ 3-5 min |
| Render | ≈ 1-3 min por clip |

## 🗺️ Roadmap

- 🎙️ Diarización de hablantes con pyannote ([`diarization.py`](clipforge/diarization.py) ya preparado) → reencuadre por hablante activo
- ⏩ Speed-ramps variables por tramo (el `TimeMap` ya soporta velocidad global)
- 🎬 B-roll automático en los picos de excitación
- 📱 Subida automática a YouTube/TikTok usando los metadatos de `clip_NNN.json`

## 📄 Licencia

[MIT](LICENSE) © 2026 Rafa Pecino · Canal [PecinoGP](https://www.youtube.com/channel/UCSvr3yH2NkqlAHfuRDphz4g) — MotoGP, WSBK y competición

*Hecho con IA, para hacer más contenido de carreras* 🏍️💨
