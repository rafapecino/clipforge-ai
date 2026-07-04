"""Dirección de efectos por clip (determinista por semilla):

- zoom base lento (in/out alternado según el clip)
- punch zooms en picos de excitación acústica y momentos con emoji
- camera shake amortiguado en el pico principal
- flash blanco breve en los 1-2 picos más fuertes
- motion blur ligero durante los punch (mezcla de fotogramas en el renderer)
"""
import numpy as np


class PlanFX:
    def __init__(self, dur, seed=0, activo=True, base_max=1.06, punch_amp=0.10):
        self.dur = dur
        self.activo = activo
        rng = np.random.RandomState(seed)
        self.dir_in = bool(rng.randint(2))
        self.base_max = base_max
        self.punch_amp = punch_amp
        self.punches, self.shakes, self.flashes = [], [], []

    def poblar(self, picos_out, emo_ts):
        """Coloca efectos en picos de audio y momentos de emoji."""
        if not self.activo:
            return self
        cands = sorted({round(t, 2) for t in list(picos_out) + list(emo_ts)
                        if 1.2 < t < self.dur - 1.2})
        sel = []
        for t in cands:
            if all(abs(t - x) >= 5.0 for x in sel):
                sel.append(t)
            if len(sel) >= 4:
                break
        self.punches = sel
        if sel:
            self.shakes = [sel[0]]
            self.flashes = sel[:2]
        return self

    def zoom(self, t):
        if not self.activo:
            return 1.0
        p = t / max(self.dur, 0.1)
        z = 1.0 + (self.base_max - 1.0) * (p if self.dir_in else (1 - p))
        for t0 in self.punches:
            d = t - t0
            if 0 <= d < 0.9:
                env = (d / 0.08) if d < 0.08 else float(np.exp(-(d - 0.08) / 0.30))
                z += self.punch_amp * env
        return min(z, 1.20)

    def dzoom(self, t):
        """Velocidad de zoom (para activar el motion blur)."""
        return abs(self.zoom(t + 0.033) - self.zoom(t)) * 30.0

    def shake(self, t):
        for t0 in self.shakes:
            d = t - t0
            if 0 <= d < 0.45:
                a = 8.0 * (1 - d / 0.45)
                return (a * np.sin(2 * np.pi * 11 * d),
                        0.6 * a * np.sin(2 * np.pi * 9 * d + 1.3))
        return (0.0, 0.0)

    def flash(self, t):
        for t0 in self.flashes:
            d = t - t0
            if 0 <= d < 0.14:
                return 0.6 * (1 - d / 0.14)
        return 0.0
