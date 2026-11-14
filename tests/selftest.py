"""Самопроверка алгоритмов на синтетическом «небе» с пылью (эталон известен).

python tests/selftest.py                 — проверка качества
python tests/selftest.py --make-sample D — создать демо-набор D/in/*.jpg + D/map.json
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dustheal.core.detect import detect_spots  # noqa: E402
from dustheal.core.heal import heal_all, to_float  # noqa: E402
from dustheal.core.spots import Spot, save_template, spots_to_template  # noqa: E402

SPOTS = [(200, 150, 8, 0.2), (520, 300, 16, 0.35), (900, 200, 24, 0.15), (350, 520, 12, 0.3)]


def make(seed=1, H=700, W=1100, shift=(0, 0), colors=((0.3, 0.5, 0.9), (0.8, 0.85, 0.95))):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = (yy / H)[:, :, None]
    clean = np.array(colors[0]) * (1 - t) + np.array(colors[1]) * t
    clean = np.clip(clean + rng.normal(0, 0.008, clean.shape), 0, 1).astype(np.float32)
    dirty = clean.copy()
    for x, y, r, s in SPOTS:
        d = np.hypot(xx - x - shift[0], yy - y - shift[1])
        prof = 1 / (1 + np.exp(np.clip((d - r * 0.8) / (r * 0.12), -50, 50)))
        dirty *= (1 - s * prof)[:, :, None]
    return clean, dirty


def check():
    clean, dirty = make()
    d16 = (dirty * 65535 + 0.5).astype(np.uint16)
    found = detect_spots(d16, 0.5)
    hits = sum(any(np.hypot(f.center[0] - x, f.center[1] - y) < r for f in found) for x, y, r, _ in SPOTS)
    print(f"Автопоиск: найдено {len(found)}, совпало с эталоном {hits}/{len(SPOTS)}")
    out = d16.copy()
    heal_all(out, [Spot([(x, y)], r * 1.3) for x, y, r, _ in SPOTS])
    ok = hits == len(SPOTS)
    for x, y, r, _ in SPOTS:
        R = int(r * 1.4)
        sl = (slice(y - R, y + R), slice(x - R, x + R))
        e0 = np.sqrt(((dirty[sl] - clean[sl]) ** 2).mean()) * 255
        e1 = np.sqrt(((to_float(out)[sl] - clean[sl]) ** 2).mean()) * 255
        print(f"  пятно ({x},{y}) r={r}: ошибка до {e0:5.1f}  после {e1:4.2f} (в единицах 8 бит)")
        ok &= e1 < 2.0
    print("РЕЗУЛЬТАТ:", "OK" if ok else "ПРОВАЛ")
    return 0 if ok else 1


def make_sample(d):
    from PIL import Image
    os.makedirs(os.path.join(d, "in"), exist_ok=True)
    for k, (sh, col) in enumerate([((0, 0), ((0.3, 0.5, 0.9), (0.8, 0.85, 0.95))),
                                   ((3, -2), ((0.9, 0.6, 0.3), (0.4, 0.3, 0.5)))]):
        _, dirty = make(k, shift=sh, colors=col)
        Image.fromarray((dirty * 255 + 0.5).astype(np.uint8)).save(os.path.join(d, "in", f"photo{k}.jpg"), quality=97)
    save_template(os.path.join(d, "map.json"),
                  spots_to_template([Spot([(x, y)], r * 1.3) for x, y, r, _ in SPOTS], 1100, 700))
    print("Демо-набор создан:", d)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--make-sample":
        sys.exit(make_sample(sys.argv[2]))
    sys.exit(check())
