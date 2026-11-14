"""Описание артефакта (пятна) и карта пятен (шаблон) для пакетной обработки."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import cv2
import numpy as np

METHODS = {
    "auto": "Авто (подбор метода)",
    "dust": "Пыль: снятие затемнения (сохраняет детали)",
    "heal": "Восстановление: текстура + Пуассон",
    "fill": "Плавная заливка (небо, вода, гладкие фоны)",
}


@dataclass
class Spot:
    points: list            # [(x, y), ...] — 1 точка = круглое пятно, больше — мазок кистью
    r: float                # радиус (px)
    method: str = "auto"
    used_method: str = ""   # фактически применённый метод (для auto)
    src_rect: tuple | None = field(default=None, compare=False)  # область-донор (x0,y0,x1,y1)

    def copy(self) -> "Spot":
        return Spot([tuple(p) for p in self.points], self.r, self.method, self.used_method, self.src_rect)

    @property
    def center(self):
        a = np.asarray(self.points, dtype=np.float64)
        return float(a[:, 0].mean()), float(a[:, 1].mean())

    def bbox(self, pad: float = 0.0):
        a = np.asarray(self.points, dtype=np.float64)
        return (a[:, 0].min() - self.r - pad, a[:, 1].min() - self.r - pad,
                a[:, 0].max() + self.r + pad, a[:, 1].max() + self.r + pad)

    def contains(self, x: float, y: float, tol: float = 0.0) -> bool:
        a = np.asarray(self.points, dtype=np.float64)
        if len(a) == 1:
            return (a[0, 0] - x) ** 2 + (a[0, 1] - y) ** 2 <= (self.r + tol) ** 2
        p = np.array([x, y])
        for i in range(len(a)):
            q0 = a[i]
            q1 = a[i + 1] if i + 1 < len(a) else a[i]
            d = q1 - q0
            t = 0.0 if not d.any() else float(np.clip(np.dot(p - q0, d) / np.dot(d, d), 0, 1))
            if np.linalg.norm(p - (q0 + t * d)) <= self.r + tol:
                return True
        return False

    def draw_mask(self, shape, x0: int, y0: int, dilate: float = 0.0) -> np.ndarray:
        """Маска пятна в координатах ROI (x0,y0 — смещение ROI)."""
        m = np.zeros(shape[:2], np.uint8)
        sh = 4
        k = 1 << sh
        r = int(round((self.r + dilate) * k))
        pts = [(int(round((x - x0) * k)), int(round((y - y0) * k))) for x, y in self.points]
        for p in pts:
            cv2.circle(m, p, r, 1, -1, cv2.LINE_8, sh)
        if len(pts) > 1:
            th = max(1, int(round(2 * (self.r + dilate))))
            pl = np.array([[(px / k, py / k) for px, py in pts]], np.float64)
            cv2.polylines(m, [np.round(pl).astype(np.int32)], False, 1, th, cv2.LINE_8)
        return m.astype(bool)


# ------------------------------------------------------------------ шаблоны
def spots_to_template(spots, width: int, height: int) -> dict:
    return {
        "app": "DustHeal",
        "version": 1,
        "ref_width": int(width),
        "ref_height": int(height),
        "spots": [
            {"pts": [[x / width, y / height] for x, y in s.points], "r": s.r / width, "method": s.method}
            for s in spots
        ],
    }


def template_to_spots(tpl: dict, width: int, height: int, rotate: str = "auto_cw") -> list:
    """rotate: none | cw | ccw | auto_cw | auto_ccw — поворот карты для снимков другой ориентации."""
    rw, rh = tpl.get("ref_width", width), tpl.get("ref_height", height)
    ref_landscape = rw >= rh
    img_landscape = width >= height
    mode = rotate
    if rotate.startswith("auto"):
        mode = "none" if ref_landscape == img_landscape else rotate.split("_")[1]
    # длинная сторона используется как масштаб радиуса
    ref_long = max(rw, rh)
    scale_r = max(width, height) / ref_long
    spots = []
    for d in tpl.get("spots", []):
        pts = []
        for u, v in d["pts"]:
            if mode == "cw":      # изображение повернуто на 90° по часовой
                u, v = 1.0 - v, u
            elif mode == "ccw":
                u, v = v, 1.0 - u
            pts.append((u * width, v * height))
        r = d["r"] * rw * scale_r
        spots.append(Spot(pts, float(r), d.get("method", "auto")))
    return spots


def save_template(path: str, tpl: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tpl, f, ensure_ascii=False, indent=1)


def load_template(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        tpl = json.load(f)
    if "spots" not in tpl:
        raise ValueError("Файл не является картой пятен DustHeal")
    return tpl
