"""Reencuadre inteligente 16:9 → 9:16 con seguimiento facial.

MediaPipe FaceDetection (con respaldo Haar de OpenCV) muestrea caras a lo
largo del clip. La cara principal se elige con histéresis (no saltar entre
caras: solo cambia si otra es claramente mayor durante varias muestras),
se rellenan huecos manteniendo el último objetivo y la trayectoria se
suaviza con un filtro gaussiano + límite de velocidad de paneo.
"""
import cv2
import numpy as np

_detector = None


def _cargar_detector():
    global _detector
    if _detector is None:
        try:
            import mediapipe as mp
            fd = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.4)
            _detector = ("mediapipe", fd)
        except Exception:
            haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            _detector = ("haar", haar)
            print("   ⚠️ MediaPipe no disponible → usando detector Haar de OpenCV")
    return _detector


def detectar_caras(frame_bgr):
    """Lista de caras [(cx, cy, w, h)] en coordenadas relativas 0-1."""
    tipo, det = _cargar_detector()
    H, W = frame_bgr.shape[:2]
    if tipo == "mediapipe":
        res = det.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        out = []
        for d in (res.detections or []):
            bb = d.location_data.relative_bounding_box
            out.append((bb.xmin + bb.width / 2, bb.ymin + bb.height / 2, bb.width, bb.height))
        return out
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    caras = det.detectMultiScale(g, 1.1, 5, minSize=(max(24, W // 25), max(24, W // 25)))
    return [((x + w / 2) / W, (y + h / 2) / H, w / W, h / H) for x, y, w, h in caras]


def calcular_ruta(src, tm, W, H, fps_muestreo=2.5, modo="auto"):
    """Trayectoria de encuadre en tiempo de SALIDA: dict(t, cx, cy, ratio)."""
    dur = tm.out_dur
    n = max(4, int(dur * fps_muestreo))
    t_out = np.linspace(0, max(dur - 0.05, 0.1), n)
    cx = np.full(n, 0.5)
    cy = np.full(n, 0.46)
    hallazgos = 0
    if modo != "centro":
        cap = cv2.VideoCapture(src)
        objetivo = None
        paciencia = 0
        for i, to in enumerate(t_out):
            cap.set(cv2.CAP_PROP_POS_MSEC, tm.out2src(float(to)) * 1000.0)
            ok, fr = cap.read()
            if not ok:
                continue
            if fr.shape[1] > 640:
                fr = cv2.resize(fr, (640, int(fr.shape[0] * 640 / fr.shape[1])))
            caras = detectar_caras(fr)
            if caras:
                hallazgos += 1
                grande = max(caras, key=lambda c: c[2] * c[3])
                if objetivo is None:
                    objetivo = grande
                else:
                    cerca = [c for c in caras if abs(c[0] - objetivo[0]) < 0.13]
                    if cerca:
                        cand = max(cerca, key=lambda c: c[2] * c[3])
                        if grande[2] * grande[3] > cand[2] * cand[3] * 1.8:
                            paciencia += 1          # otra cara domina: esperar
                            if paciencia >= 3:
                                cand, paciencia = grande, 0
                        else:
                            paciencia = 0
                        objetivo = cand
                    else:
                        objetivo = grande
            if objetivo is not None:
                cx[i], cy[i] = objetivo[0], objetivo[1]
        cap.release()
    if n > 5:                                   # suavizado + límite de paneo
        sig = max(1, int(0.8 * fps_muestreo))
        ker = np.exp(-0.5 * (np.arange(-3 * sig, 3 * sig + 1) / sig) ** 2)
        ker /= ker.sum()
        for arr in (cx, cy):
            pad = np.pad(arr, 3 * sig, mode="edge")
            arr[:] = np.convolve(pad, ker, mode="same")[3 * sig:-3 * sig]
        vmax = 0.10 / fps_muestreo              # máx. 10% del ancho por segundo
        for i in range(1, n):
            cx[i] = cx[i - 1] + np.clip(cx[i] - cx[i - 1], -vmax, vmax)
            cy[i] = cy[i - 1] + np.clip(cy[i] - cy[i - 1], -vmax, vmax)
    return {"t": t_out, "cx": cx * W, "cy": cy * H, "ratio": hallazgos / n}
