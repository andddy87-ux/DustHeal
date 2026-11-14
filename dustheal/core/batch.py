"""Пакетная обработка: одна карта пятен -> множество снимков.

Для каждого снимка параметры фона (цвет, яркость, контраст, зерно) берутся из ЕГО окружения пятна,
поэтому одна карта корректно работает на разных сюжетах и экспозициях.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .detect import detect_spots, refine_spot
from .heal import heal_all
from .imageio import load_image, save_image
from .spots import template_to_spots


@dataclass
class BatchOptions:
    out_dir: str = ""
    suffix: str = "_clean"
    out_format: str = "same"        # same | .jpg | .tif | .png
    jpeg_quality: int = 97
    keep_metadata: bool = True
    overwrite: bool = False
    refine: bool = True             # уточнять положение каждого пятна на каждом снимке
    search_frac: float = 0.6
    skip_invisible: bool = False    # не трогать пятна, не найденные на данном снимке
    autodetect: bool = False        # дополнительно искать новые пятна
    sensitivity: float = 0.5
    rotate: str = "auto_cw"         # none | cw | ccw | auto_cw | auto_ccw
    force_method: str = ""          # "" = как в карте
    margin_frac: float = 0.25
    workers: int = 2


def output_path(src: str, opt: BatchOptions, tag_ext: bool = False) -> str:
    base, ext = os.path.splitext(os.path.basename(src))
    if tag_ext:  # одинаковые имена с разными расширениями: фото.jpg / фото.tif
        base += "_" + ext.lstrip(".").lower()
    out_ext = ext if opt.out_format == "same" else opt.out_format
    if out_ext.lower() == ".bmp":
        out_ext = ".png"
    out_dir = opt.out_dir or os.path.join(os.path.dirname(src), "cleaned")
    return os.path.join(out_dir, base + opt.suffix + out_ext)


def process_file(path: str, template: dict, opt: BatchOptions, out: str | None = None) -> tuple[str, str]:
    """Возвращает (путь результата, текст отчёта)."""
    out = out or output_path(path, opt)
    if os.path.abspath(out) == os.path.abspath(path) and not opt.overwrite:
        raise IOError("Результат перезаписал бы исходный файл — измените суффикс или папку")
    if os.path.exists(out) and not opt.overwrite:
        return out, "пропущен (результат уже существует)"
    img = load_image(path)
    W, H = img.width, img.height
    spots = template_to_spots(template, W, H, opt.rotate) if template else []
    if opt.force_method:
        for s in spots:
            s.method = opt.force_method
    n_tpl = len(spots)
    skipped = 0
    if opt.refine or opt.skip_invisible:
        refined = []
        for s in spots:
            ns, visible = refine_spot(img.data, s, opt.search_frac)
            if not visible and opt.skip_invisible:
                skipped += 1
                continue
            refined.append(ns if (opt.refine and visible) else s)
        spots = refined
    n_new = 0
    if opt.autodetect:
        found = detect_spots(img.data, opt.sensitivity)
        for f in found:
            fx, fy = f.center
            if all(math.hypot(fx - s.center[0], fy - s.center[1]) > (s.r + f.r) for s in spots):
                if opt.force_method:
                    f.method = opt.force_method
                spots.append(f)
                n_new += 1
    data = img.data.copy()
    heal_all(data, spots, opt.margin_frac)
    out_ext = os.path.splitext(out)[1].lower()
    if data.dtype.itemsize == 2 and out_ext in (".webp", ".bmp"):
        out = os.path.splitext(out)[0] + ".tif"
    save_image(out, img, data, opt.jpeg_quality, opt.keep_metadata)
    methods = {}
    for s in spots:
        methods[s.used_method] = methods.get(s.used_method, 0) + 1
    rep = f"пятен: {len(spots)} (из карты {n_tpl - skipped}"
    if skipped:
        rep += f", не обнаружено/пропущено {skipped}"
    if n_new:
        rep += f", найдено новых {n_new}"
    rep += ") " + ", ".join(f"{k}:{v}" for k, v in sorted(methods.items()))
    return out, rep


def run_batch(files, template, opt: BatchOptions, on_progress=None, should_stop=None):
    """on_progress(done, total, path, ok, message)"""
    total = len(files)
    done = 0
    plain = [os.path.normcase(os.path.abspath(output_path(f, opt))) for f in files]
    outs = {f: output_path(f, opt, tag_ext=plain.count(p) > 1) for f, p in zip(files, plain)}
    results = []
    with ThreadPoolExecutor(max_workers=max(1, opt.workers)) as ex:
        futs = {}
        it = iter(files)

        def submit_next():
            try:
                p = next(it)
            except StopIteration:
                return False
            futs[ex.submit(process_file, p, template, opt, outs[p])] = p
            return True

        for _ in range(max(1, opt.workers)):
            submit_next()
        while futs:
            fut = next(as_completed(list(futs.keys())))
            p = futs.pop(fut)
            done += 1
            try:
                out, rep = fut.result()
                ok, msg = True, f"{os.path.basename(out)} — {rep}"
                results.append((p, True, out))
            except Exception as e:  # noqa: BLE001
                ok, msg = False, str(e)
                results.append((p, False, msg))
            if on_progress:
                try:  # сбой вывода отчёта не должен считаться ошибкой обработки
                    on_progress(done, total, p, ok, msg)
                except Exception:  # noqa: BLE001
                    pass
            if should_stop and should_stop():
                for f in futs:
                    f.cancel()
                break
            submit_next()
    return results
