"""[ROADMAP] Diarización de hablantes con pyannote.audio.

Interfaz preparada: con un HF_TOKEN (huggingface.co) se podrá etiquetar
cada palabra con su hablante para (a) reencuadrar a quien habla en cada
momento —no solo a la cara más grande— y (b) puntuar mejor los debates.
La integración está desacoplada: ningún módulo depende de esta función.
"""


def diarizar(wav, hf_token=None):
    if not hf_token:
        raise NotImplementedError(
            "Configura HF_TOKEN en los Secretos de Colab y descomenta la "
            "integración pyannote para activar la diarización.")
    # from pyannote.audio import Pipeline
    # pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
    #                                 use_auth_token=hf_token)
    # return pipe(wav)
