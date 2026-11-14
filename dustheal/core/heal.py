"""Алгоритмы удаления пятен.

Методы:
  dust — пыль на матрице/линзе — это полупрозрачная тень (мультипликативное ослабление света).
         Оцениваем «ожидаемый» фон гармонической интерполяцией (мембрана Лапласа) низких частот
         с окружения и делим на наблюдаемое. Детали и зерно внутри пятна сохраняются.
  heal — восстанавливающая кисть: автоматически ищется лучший донорский участок рядом
         (совпадение текстуры, шума и структуры кольца), затем бесшовная вставка по Пуассону:
         цвет, яркость и контраст подстраиваются под окружение пятна.
  fill — плавная заливка: гладкая мембрана по краю (идеально для градиентов неба/воды)
         + зерно/шум, перенесённый с подходящего донора — без «пластикового» эффекта.
  auto — выбор метода по анализу пятна и окружения.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .spots import Spot

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


# --------------------------------------------------------------- утилиты
def to_float(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint8:
        return a.astype(np.float32) * (1.0 / 255.0)
    if a.dtype == np.uint16:
        return a.astype(np.float32) * (1.0 / 65535.0)
    return a.astype(np.float32)


def from_float(f: np.ndarray, dtype) -> np.ndarray:
    if dtype == np.uint8:
        return np.clip(f * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if dtype == np.uint16:
        return np.clip(f * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    return f.astype(dtype)


def _ellipse(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def dilate(mask: np.ndarray, radius: float) -> np.ndarray:
    if radius < 0.5:
        return mask.copy()
    return cv2.dilate(mask.astype(np.uint8), _ellipse(int(math.ceil(radius)))).astype(bool)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def harmonic_fill(field: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Решение уравнения Лапласа внутри omega с граничными значениями field снаружи.

    Даёт максимально гладкую поверхность, точно совпадающую с краем (мембранная интерполяция).
    Для больших областей решение ищется на уменьшенной сетке и уточняется.
    """
    single = field.ndim == 2
    F = field[:, :, None] if single else field
    n = int(omega.sum())
    if n == 0:
        return field.copy()
    if n > 120_000:
        out = _harmonic_multiscale(F, omega)
    else:
        out = _harmonic_direct(F, omega)
    return out[:, :, 0] if single else out


def _harmonic_direct(F, omega, init=None):
    H, W = omega.shape
    ys, xs = np.nonzero(omega)
    n = len(ys)
    idx = np.full((H, W), -1, np.int64)
    idx[ys, xs] = np.arange(n)
    C = F.shape[2]
    rhs = np.zeros((n, C), np.float64)
    diag = np.zeros(n, np.float64)
    rows, cols = [], []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = ys + dy, xs + dx
        valid = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        diag += valid
        src = np.nonzero(valid)[0]
        nyv, nxv = ny[valid], nx[valid]
        nid = idx[nyv, nxv]
        inside = nid >= 0
        rows.append(src[inside])
        cols.append(nid[inside])
        o = ~inside
        np.add.at(rhs, src[o], F[nyv[o], nxv[o]].astype(np.float64))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    A = sparse.csc_matrix((np.full(len(rows), -1.0), (rows, cols)), shape=(n, n))
    A = A + sparse.diags(np.maximum(diag, 1e-9))
    try:
        sol = splu(A.tocsc(), permc_spec="COLAMD").solve(rhs)
    except RuntimeError:
        # изолированная область без границы (не должно случаться) — среднее значение
        sol = np.repeat(F[~omega].mean(axis=0, keepdims=True), n, axis=0)
    out = F.astype(np.float32).copy()
    out[ys, xs] = sol.astype(np.float32)
    return out


def _harmonic_multiscale(F, omega):
    """Грубое решение на сетке /4 + точное решение в узкой полосе у границы + сглаживание Якоби."""
    H, W = omega.shape
    f = 4
    small = cv2.resize(F, (max(2, W // f), max(2, H // f)), interpolation=cv2.INTER_AREA)
    if small.ndim == 2:
        small = small[:, :, None]
    om_s = cv2.resize(omega.astype(np.uint8), (small.shape[1], small.shape[0]),
                      interpolation=cv2.INTER_NEAREST).astype(bool)
    om_s = cv2.erode(om_s.astype(np.uint8), _ellipse(1)).astype(bool)
    sol_s = harmonic_fill(small, om_s)
    if sol_s.ndim == 2:
        sol_s = sol_s[:, :, None]
    up = cv2.resize(sol_s, (W, H), interpolation=cv2.INTER_CUBIC)
    if up.ndim == 2:
        up = up[:, :, None]
    out = F.astype(np.float32).copy()
    out[omega] = up[omega]
    # итерации Якоби для согласования с точной границей
    om = omega
    k = np.array([[0, .25, 0], [.25, 0, .25], [0, .25, 0]], np.float32)
    for _ in range(60):
        avg = cv2.filter2D(out, -1, k, borderType=cv2.BORDER_REPLICATE)
        if avg.ndim == 2:
            avg = avg[:, :, None]
        out[om] = avg[om]
    return out


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# --------------------------------------------------------------- геометрия пятна
class _Geom:
    def __init__(self, img_shape, spot: Spot, margin_frac: float):
        H, W = img_shape[:2]
        r = max(1.0, spot.r)
        self.r = r
        self.m = max(2.0, r * margin_frac)                 # запас на мягкий ореол пятна
        self.sig = float(np.clip(r * 0.12, 0.8, 6.0))      # масштаб низких частот
        self.bpad = math.ceil(2.5 * self.sig) + 1          # защита от «затекания» пятна при размытии
        self.ring = int(max(4, round(r * 0.7)))            # ширина кольца-образца
        pad = self.m + self.bpad + self.ring + 4
        bx0, by0, bx1, by1 = spot.bbox(pad)
        self.x0, self.y0 = max(0, int(math.floor(bx0))), max(0, int(math.floor(by0)))
        self.x1, self.y1 = min(W, int(math.ceil(bx1)) + 1), min(H, int(math.ceil(by1)) + 1)
        shp = (self.y1 - self.y0, self.x1 - self.x0)
        self.core = spot.draw_mask(shp, self.x0, self.y0)
        self.ext = spot.draw_mask(shp, self.x0, self.y0, dilate=self.m)
        self.omega = dilate(self.ext, self.bpad)
        ring = dilate(self.omega, self.ring) & ~self.omega
        self.ringmask = ring
        dt = cv2.distanceTransform(self.omega.astype(np.uint8), cv2.DIST_L2, 5)
        self.alpha = _smoothstep(dt / max(1.0, self.m + self.bpad))[:, :, None]


# --------------------------------------------------------------- поиск донора
def find_source(imgF_getter, img_shape, g: _Geom, T: np.ndarray, forbidden_rects):
    """Перебор смещений по кольцам вокруг пятна; оценка совпадения окружения и текстуры."""
    H, W = img_shape[:2]
    h, w = g.omega.shape
    use = g.omega | g.ringmask
    ext_r = 0.5 * math.hypot(h, w)
    dmin = max(2 * (g.r + g.m + g.bpad) + 2, ext_r * 0.9)
    dmax = dmin * 3.2 + 10
    # окно поиска
    wx0 = int(max(0, g.x0 - dmax - 2)); wy0 = int(max(0, g.y0 - dmax - 2))
    wx1 = int(min(W, g.x1 + dmax + 2)); wy1 = int(min(H, g.y1 + dmax + 2))
    win = imgF_getter(wx0, wy0, wx1, wy1)
    winB = blur(win, 1.0)
    winHP = (win - winB) @ LUMA
    forb = np.zeros(win.shape[:2], bool)
    for (fx0, fy0, fx1, fy1) in forbidden_rects:
        a0, b0 = int(max(0, fx0 - wx0)), int(max(0, fy0 - wy0))
        a1, b1 = int(min(win.shape[1], fx1 - wx0)), int(min(win.shape[0], fy1 - wy0))
        if a1 > a0 and b1 > b0:
            forb[b0:b1, a0:a1] = True

    Tb = blur(T, 1.0)
    rng = np.random.default_rng(0)

    def _sample(mask, nmax):
        yy, xx = np.nonzero(mask)
        if len(yy) > nmax:
            sel = rng.choice(len(yy), nmax, replace=False)
            yy, xx = yy[sel], xx[sel]
        return yy, xx

    ry, rx = _sample(g.ringmask, 3000)
    oy_, ox_ = _sample(g.omega, 3000)
    uy, ux = _sample(use, 6000)
    Tr = Tb[ry, rx]
    hpT = ((T - Tb) @ LUMA)[ry, rx]
    stdT = float(hpT.std()) + 1e-4
    p98T = float(np.percentile(np.abs(hpT), 98)) + 1e-4
    lumR = Tr @ LUMA
    contrastT = float(lumR.std()) + 1e-3
    q98 = max(1, int(len(oy_) * 0.02))

    def score(ox, oy):
        sx0, sy0 = g.x0 + ox - wx0, g.y0 + oy - wy0
        if sx0 < 0 or sy0 < 0 or sx0 + w > win.shape[1] or sy0 + h > win.shape[0]:
            return None
        if forb[uy + sy0, ux + sx0].any():
            return None
        if abs(ox) < w and abs(oy) < h:  # донор не должен перекрывать само пятно
            a = g.omega[max(0, oy):h + min(0, oy), max(0, ox):w + min(0, ox)]
            b = g.ext[max(0, -oy):h + min(0, -oy), max(0, -ox):w + min(0, -ox)]
            if (a & b).any():
                return None
        d = Tr - winB[ry + sy0, rx + sx0]
        mu = d.mean(axis=0)
        s_struct = float(((d - mu) ** 2).mean())
        s_off = float((mu ** 2).sum())
        hpS = winHP[oy_ + sy0, ox_ + sx0]
        s_tex = (float(hpS.std()) - stdT) ** 2
        top = float(np.partition(np.abs(hpS), -q98)[-q98])
        s_out = max(0.0, top - 1.6 * p98T) ** 2
        # структура внутри донора не должна сильно отличаться от структуры кольца
        lumS = winB[oy_ + sy0, ox_ + sx0] @ LUMA
        s_int = max(0.0, float(lumS.std()) - 1.5 * contrastT) ** 2
        dist = math.hypot(ox, oy)
        return s_struct + 0.25 * s_off + 3.0 * s_tex + 4.0 * s_out + 2.0 * s_int + 1e-9 * dist

    cands = []
    for d in np.linspace(dmin, dmax, 7):
        n_ang = 24 if d < dmin * 2 else 32
        for a in np.arange(n_ang) * (2 * math.pi / n_ang):
            ox, oy = int(round(d * math.cos(a))), int(round(d * math.sin(a)))
            s = score(ox, oy)
            if s is not None:
                cands.append((s, ox, oy))
    if not cands:
        return None
    cands.sort()
    best = cands[0]
    step = max(1, int(dmin / 10))
    for s0, ox0, oy0 in cands[:4]:
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                if dx == 0 and dy == 0:
                    continue
                s = score(ox0 + dx * step, oy0 + dy * step)
                if s is not None and s < best[0]:
                    best = (s, ox0 + dx * step, oy0 + dy * step)
    _, ox, oy = best
    sx0, sy0 = g.x0 + ox - wx0, g.y0 + oy - wy0
    S = win[sy0:sy0 + h, sx0:sx0 + w].copy()
    return S, (g.x0 + ox, g.y0 + oy, g.x1 + ox, g.y1 + oy)


# --------------------------------------------------------------- методы
def _method_dust(T, g: _Geom):
    """Оценка карты ослабления A = наблюдаемое / ожидаемое на мелком масштабе,
    затем билатеральное сглаживание: убирает влияние текстуры, но сохраняет резкий край тени
    (при закрытой диафрагме пыль даёт чёткие края). Результат = T / A."""
    s0 = 0.7
    om = dilate(g.ext, 2)
    L = blur(T, s0)
    E = harmonic_fill(L, om)
    A0 = np.clip(L / np.maximum(E, 1e-4), 0.05, 2.0).astype(np.float32)
    lum = L @ LUMA
    T1 = blur(T, 1.0) @ LUMA
    tex = float((T1 - blur(T1 * 1.0, 4.0))[g.ringmask].std())
    # на пёстром фоне (трава, листва) карту ослабления сглаживаем сильнее, чтобы не «съесть» текстуру
    sS = max(1.0, g.r * (0.12 if tex < 0.012 else 0.35))
    fl = lum / np.maximum(blur(lum, sS), 1e-4)
    na = float(fl[g.ringmask].std()) + 1e-3          # уровень «текстурных» колебаний
    if sS > 6.0:  # крупное пятно: фильтруем на уменьшенной копии (быстро, результат тот же)
        q = 6.0 / sS
        hs, ws = A0.shape[:2]
        As = cv2.resize(A0, (max(2, int(ws * q)), max(2, int(hs * q))), interpolation=cv2.INTER_AREA)
        As = cv2.bilateralFilter(As, 25, 10.0 * na, 6.0)
        A = cv2.resize(As, (ws, hs), interpolation=cv2.INTER_CUBIC)
    else:
        A = cv2.bilateralFilter(A0, int(2 * round(2 * sS) + 1), 10.0 * na, sS)
    A[~om] = 1.0
    G = 1.0 / np.maximum(A, 0.05)
    return T * G, om, G


def _grain_from_ring(T, g, sig):
    """Резерв: зерно из пикселей кольца (если донор не найден)."""
    hp = T - blur(T, sig)
    pool = hp[g.ringmask]
    rng = np.random.default_rng(12345)
    out = np.zeros_like(T)
    if len(pool):
        out[g.omega] = pool[rng.integers(0, len(pool), int(g.omega.sum()))]
    return blur(out, 0.35)


def _method_fill(T, g: _Geom, S):
    sig = 1.6
    L = blur(T, sig)
    E = harmonic_fill(L, g.omega)
    grain = (S - blur(S, sig)) if S is not None else _grain_from_ring(T, g, sig)
    return E + grain


def _method_heal(T, g: _Geom, S):
    D = harmonic_fill(blur(T, 1.0) - blur(S, 1.0), g.omega)
    return S + D


def _analyze(T, g: _Geom):
    """Возвращает (метод, сведения) для режима auto."""
    out, om, G = _method_dust(T, g)
    Gl = G @ LUMA / LUMA.sum()
    core = g.core
    gc = Gl[core]
    gmed = float(np.median(gc) - 1.0)
    gmax = float(gc.max())
    gmin = float(gc.min())
    Tb = blur(T, 1.0)
    hp = (T - Tb) @ LUMA
    ring_std = float(hp[g.ringmask].std())
    in_std = float((hp * Gl)[core].std())
    ratio = (in_std + 2e-3) / (ring_std + 2e-3)
    mid = (Tb - blur(T, 4.0)) @ LUMA
    ring_tex = float(mid[g.ringmask].std())
    info = dict(gmed=gmed, gmax=gmax, ratio=ratio, ring_tex=ring_tex, ring_std=ring_std)
    darker = gmed > 0.004
    transparent = gmax < 2.6 and gmin > (0.85 if ring_tex < 0.012 else 0.6) and 0.55 < ratio < 1.8
    if darker and transparent and ring_tex < 0.04:
        return "dust", info
    if ring_tex < 0.012:
        return "fill", info
    return "heal", info


def heal_spot(img: np.ndarray, spot: Spot, forbidden_rects=(), margin_frac: float = 0.25,
              method: str | None = None) -> tuple:
    """Удалить пятно на изображении img (H x W x 3, uint8/uint16/float) — изменяется на месте.

    forbidden_rects — прямоугольники (x0,y0,x1,y1) других пятен, которые нельзя брать как донор.
    Возвращает изменённый прямоугольник (x0, y0, x1, y1).
    """
    method = method or spot.method or "auto"
    g = _Geom(img.shape, spot, margin_frac)
    if g.x1 - g.x0 < 3 or g.y1 - g.y0 < 3:
        return (0, 0, 0, 0)
    roi = img[g.y0:g.y1, g.x0:g.x1]
    T = to_float(roi)

    if method == "auto":
        method, _ = _analyze(T, g)
    spot.used_method = method
    spot.src_rect = None

    if method == "dust":
        out, om, _ = _method_dust(T, g)
        result = out
        alpha = om[:, :, None].astype(np.float32)
    else:
        getter = lambda x0, y0, x1, y1: to_float(img[y0:y1, x0:x1])  # noqa: E731
        src = find_source(getter, img.shape, g, T, list(forbidden_rects))
        S = None
        if src is not None:
            S, spot.src_rect = src
        if method == "heal" and S is None:
            method = "fill"
            spot.used_method = method
        result = _method_heal(T, g, S) if method == "heal" else _method_fill(T, g, S)
        alpha = g.alpha
    blended = alpha * result + (1.0 - alpha) * T
    roi[...] = from_float(np.clip(blended, 0.0, 1.0), img.dtype)
    return (g.x0, g.y0, g.x1, g.y1)


def spot_rect(spot: Spot, margin_frac: float = 0.25):
    r = max(1.0, spot.r)
    m = max(2.0, r * margin_frac) + math.ceil(2.5 * float(np.clip(r * 0.12, 0.8, 6.0))) + 2
    return spot.bbox(m)


def heal_all(img: np.ndarray, spots, margin_frac: float = 0.25, progress=None) -> np.ndarray:
    """Применить все пятна последовательно (img изменяется на месте и возвращается)."""
    rects = [spot_rect(s, margin_frac) for s in spots]
    for i, s in enumerate(spots):
        # донором нельзя брать ещё не обработанные пятна; уже исправленные — можно
        forb = rects[i + 1:]
        heal_spot(img, s, forb, margin_frac)
        if progress:
            progress(i + 1, len(spots))
    return img
