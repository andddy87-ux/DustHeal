"""Автоматический поиск пятен от пыли/грязи и уточнение их положения на других снимках."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .heal import LUMA, _ellipse, blur, to_float
from .spots import Spot


def _log_luma(rgbF: np.ndarray) -> np.ndarray:
    return np.log(rgbF @ LUMA + 0.01).astype(np.float32)


def _texture_map(Y: np.ndarray, sigma: float) -> np.ndarray:
    """Локальная «занятость» фона: мелкая детализация (пыль мягкая и её почти не даёт)."""
    lap = cv2.Laplacian(blur(Y, 1.0), cv2.CV_32F, ksize=3)
    return blur(np.abs(lap), sigma)


def detect_spots(img: np.ndarray, sensitivity: float = 0.5, min_radius: float = 2.0,
                 max_radius: float | None = None, smooth_only: bool = True,
                 dark_only: bool = True) -> list:
    """Найти пятна. sensitivity 0..1 (больше — находит более слабые пятна).

    Возвращает список Spot в координатах исходного изображения.
    """
    H, W = img.shape[:2]
    long_side = max(H, W)
    f = min(1.0, 4000.0 / long_side)
    small = cv2.resize(img, (max(8, int(W * f)), max(8, int(H * f))), interpolation=cv2.INTER_AREA) if f < 1 else img
    Y = _log_luma(to_float(small))
    Ys = blur(Y, 1.0)
    if max_radius is None:
        max_radius = long_side * 0.02
    rmax = max(3.0, max_radius * f)
    rmin = max(1.0, min_radius * f)

    # многомасштабная оценка фона: морфологическое закрытие убирает тёмные пятна меньше ядра.
    # Крупные масштабы считаются на уменьшенной копии (пирамида) — это быстро.
    D = np.zeros_like(Ys)
    hs, ws = Ys.shape
    rs = rmin
    while True:
        rr = min(rs, rmax)
        q = min(1.0, 6.0 / rr)
        Yq = Ys if q >= 1 else cv2.resize(Ys, (max(4, int(ws * q)), max(4, int(hs * q))), interpolation=cv2.INTER_AREA)
        kk = max(1, int(round(rr * q * 1.3)))
        bg = cv2.morphologyEx(Yq, cv2.MORPH_CLOSE, _ellipse(kk), borderType=cv2.BORDER_REFLECT)
        bg = blur(bg, max(0.8, kk / 2.5))
        if not dark_only:
            bgo = cv2.morphologyEx(Yq, cv2.MORPH_OPEN, _ellipse(kk), borderType=cv2.BORDER_REFLECT)
            bgo = blur(bgo, max(0.8, kk / 2.5))
        if q < 1:
            bg = cv2.resize(bg, (ws, hs), interpolation=cv2.INTER_LINEAR)
            if not dark_only:
                bgo = cv2.resize(bgo, (ws, hs), interpolation=cv2.INTER_LINEAR)
        D = np.maximum(D, bg - Ys)
        if not dark_only:
            D = np.maximum(D, Ys - bgo)
        if rr >= rmax:
            break
        rs *= 2.0

    tex = _texture_map(Y, max(3.0, rmax * 0.6))
    s = float(np.clip(sensitivity, 0.0, 1.0))
    thr = 0.09 * (1 - s) ** 1.6 + 0.007          # минимальное относительное затемнение (~0.7%..9%)
    tex_k = 1.2 + (1 - s) * 3.0                  # насколько пятно должно выделяться над текстурой
    noise = float(np.median(tex)) + 1e-4

    Db = blur(D, 0.8)
    binary = (Db > thr) & (Db > tex_k * np.maximum(tex, noise) * 0.35)
    binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, _ellipse(1))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)

    spots = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < max(3, math.pi * rmin * rmin * 0.5):
            continue
        req = math.sqrt(area / math.pi)
        if req > rmax * 1.2:
            continue
        ar = max(w, h) / max(1, min(w, h))
        fillr = area / float(w * h)
        if ar > 2.6 or fillr < 0.45:
            continue
        sub = lab[y:y + h, x:x + w] == i
        peak = float(Db[y:y + h, x:x + w][sub].max())
        if smooth_only:
            # окружение пятна должно быть достаточно гладким (небо, вода, стена...)
            pad = int(max(3, req * 1.5))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(Y.shape[1], x + w + pad), min(Y.shape[0], y + h + pad)
            ring = np.ones((y1 - y0, x1 - x0), bool)
            ring[y - y0:y - y0 + h, x - x0:x - x0 + w] &= ~sub
            ring_tex = float(np.median(tex[y0:y1, x0:x1][ring]))
            if peak < tex_k * ring_tex * 1.4:
                continue
            # пятно не должно быть частью протяжённого тёмного объекта
            ring_d = float(np.percentile(Db[y0:y1, x0:x1][ring], 75))
            if ring_d > peak * 0.6:
                continue
        cx, cy = cents[i]
        r = (req * 1.35 + 1.0) / f
        spots.append((peak, Spot([(float(cx / f), float(cy / f))], float(max(r, min_radius)), "auto")))
    spots.sort(key=lambda t: -t[0])
    return [s for _, s in spots]


def refine_spot(img: np.ndarray, spot: Spot, search_frac: float = 0.6,
                min_contrast: float = 0.006) -> tuple:
    """Уточнить центр/радиус круглого пятна на конкретном снимке.

    Возвращает (spot, visible). visible=False — артефакт на этом снимке не обнаружен
    (например, пятно попало на сложную текстуру или открытая диафрагма его «растворила»).
    """
    if len(spot.points) != 1:
        return spot, True
    H, W = img.shape[:2]
    cx, cy = spot.points[0]
    r = spot.r
    search = max(3.0, r * search_frac)
    pad = int(r * 2.5 + search + 6)
    x0, y0 = max(0, int(cx - pad)), max(0, int(cy - pad))
    x1, y1 = min(W, int(cx + pad) + 1), min(H, int(cy + pad) + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return spot, False
    Y = blur(_log_luma(to_float(img[y0:y1, x0:x1])), max(0.8, r * 0.1))
    kk = int(r * 2.6) + 2
    bg = blur(cv2.morphologyEx(Y, cv2.MORPH_CLOSE, _ellipse(kk)), max(1.0, kk / 3))
    D = np.maximum(bg - Y, 0)
    Db = blur(D, max(0.8, r / 3))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    within = (xx - cx) ** 2 + (yy - cy) ** 2 <= search ** 2
    if not within.any():
        return spot, False
    vals = np.where(within, Db, -1)
    iy, ix = np.unravel_index(int(np.argmax(vals)), vals.shape)
    peak = float(Db[iy, ix])
    # контраст относительно шумового фона окна
    noise = float(np.percentile(Db, 50))
    if peak < min_contrast or peak < noise * 3:
        return spot, False
    comp = (Db > peak * 0.45).astype(np.uint8)
    n, lab = cv2.connectedComponents(comp)
    sub = lab == lab[iy, ix]
    wts = Db * sub
    sw = float(wts.sum())
    ncx = float((wts * xx).sum() / sw)
    ncy = float((wts * yy).sum() / sw)
    if math.hypot(ncx - cx, ncy - cy) > search:
        ncx, ncy = float(xx[iy, ix]), float(yy[iy, ix])
    req = math.sqrt(sub.sum() / math.pi) * 1.6 + 1
    nr = float(np.clip(req, r * 0.75, r * 1.4))
    return Spot([(ncx, ncy)], max(nr, r), spot.method), True
