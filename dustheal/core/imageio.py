"""Чтение/запись изображений с поддержкой Unicode-путей Windows, 8/16 бит и метаданных."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

try:
    import tifffile
except Exception:  # pragma: no cover
    tifffile = None

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


@dataclass
class LoadedImage:
    data: np.ndarray                 # H x W x 3, RGB, uint8 или uint16
    path: str = ""
    alpha: np.ndarray | None = None  # H x W, тот же dtype
    gray: bool = False
    exif: bytes | None = None
    icc: bytes | None = None
    extra: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def bit_depth(self) -> int:
        return 16 if self.data.dtype == np.uint16 else 8


def _normalize_dtype(arr: np.ndarray) -> np.ndarray:
    if arr.dtype in (np.uint8, np.uint16):
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        mx = float(np.nanmax(arr)) if arr.size else 1.0
        scale = 65535.0 if mx <= 1.0 + 1e-6 else 65535.0 / max(mx, 1e-9)
        return np.clip(np.nan_to_num(arr) * scale + 0.5, 0, 65535).astype(np.uint16)
    if arr.dtype == np.int16:
        return (arr.astype(np.int32) + 32768).astype(np.uint16)
    if arr.dtype in (np.uint32, np.int32):
        return np.clip(arr >> 16, 0, 65535).astype(np.uint16)
    return arr.astype(np.uint16)


def load_image(path: str) -> LoadedImage:
    buf = np.fromfile(path, dtype=np.uint8)
    arr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    from_cv = arr is not None
    if arr is None and tifffile is not None and path.lower().endswith((".tif", ".tiff")):
        arr = tifffile.imread(path)
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[2] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)
    if arr is None:
        raise IOError(f"Не удалось прочитать изображение: {path}")

    arr = _normalize_dtype(arr)
    alpha = None
    gray = False
    if arr.ndim == 2:
        gray = True
        rgb = np.dstack([arr, arr, arr])
    else:
        if arr.shape[2] == 4:
            alpha = arr[:, :, 3].copy()
            arr = arr[:, :, :3]
        elif arr.shape[2] == 2:  # gray + alpha
            alpha = arr[:, :, 1].copy()
            arr = np.dstack([arr[:, :, 0]] * 3)
            gray = True
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) if from_cv else arr[:, :, :3]
    rgb = np.ascontiguousarray(rgb)

    exif = icc = None
    try:
        with Image.open(path) as im:
            icc = im.info.get("icc_profile")
            if not path.lower().endswith((".tif", ".tiff")):
                exif = im.info.get("exif")
    except Exception:
        pass
    if icc is None and tifffile is not None and path.lower().endswith((".tif", ".tiff")):
        try:
            with tifffile.TiffFile(path) as tf:
                tag = tf.pages[0].tags.get(34675)
                if tag is not None:
                    icc = bytes(tag.value)
        except Exception:
            pass
    return LoadedImage(rgb, path, alpha, gray, exif, icc)


def _compose(img: LoadedImage, data: np.ndarray, keep_alpha: bool) -> np.ndarray:
    out = data[:, :, 0] if img.gray else data
    if keep_alpha and img.alpha is not None:
        a = img.alpha
        if a.dtype != out.dtype:
            a = (a.astype(np.uint32) * 257).astype(np.uint16) if out.dtype == np.uint16 else (a >> 8).astype(np.uint8)
        out = np.dstack([out, a]) if out.ndim == 3 else np.dstack([out, a])
    return out


def save_image(path: str, img: LoadedImage, data: np.ndarray | None = None,
               jpeg_quality: int = 97, keep_metadata: bool = True) -> str:
    """Сохранить результат. data — обработанный массив (по умолчанию img.data)."""
    data = img.data if data is None else data
    ext = os.path.splitext(path)[1].lower()
    exif = img.exif if keep_metadata else None
    icc = img.icc if keep_metadata else None
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    if ext in (".jpg", ".jpeg"):
        d8 = data if data.dtype == np.uint8 else ((data.astype(np.uint32) + 128) // 257).astype(np.uint8)
        arr = d8[:, :, 0] if img.gray else d8
        pil = Image.fromarray(arr)
        kw = dict(quality=int(jpeg_quality), subsampling=0, optimize=True)
        if exif:
            kw["exif"] = exif
        if icc:
            kw["icc_profile"] = icc
        pil.save(path, "JPEG", **kw)
        return path

    if data.dtype == np.uint8 and ext in (".png", ".tif", ".tiff", ".webp", ".bmp"):
        arr = _compose(img, data, keep_alpha=ext != ".bmp")
        pil = Image.fromarray(arr)
        kw = {}
        if icc and ext != ".bmp":
            kw["icc_profile"] = icc
        if exif and ext in (".png", ".webp"):
            kw["exif"] = exif
        if ext in (".tif", ".tiff"):
            kw["compression"] = "tiff_lzw"
            pil.save(path, "TIFF", **kw)
        elif ext == ".png":
            pil.save(path, "PNG", compress_level=6, **kw)
        elif ext == ".webp":
            pil.save(path, "WEBP", lossless=True, **kw)
        else:
            pil.save(path, "BMP")
        return path

    # 16 бит
    arr = _compose(img, data, keep_alpha=True)
    if ext in (".tif", ".tiff") and tifffile is not None:
        photometric = "minisblack" if arr.ndim == 2 else "rgb"
        extras = {"extrasamples": [2]} if (arr.ndim == 3 and arr.shape[2] == 4) else {}
        try:
            tifffile.imwrite(path, arr, photometric=photometric, compression="zlib",
                             iccprofile=icc if icc else None, **extras)
        except TypeError:
            tifffile.imwrite(path, arr, photometric=photometric, compression="zlib", **extras)
        return path
    if ext in (".webp", ".bmp"):
        raise IOError("Формат не поддерживает 16 бит — выберите TIFF или PNG")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA if arr.shape[2] == 4 else cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(ext if ext in (".png", ".tif", ".tiff") else ".png", arr)
    if not ok:
        raise IOError(f"Ошибка кодирования {path}")
    enc.tofile(path)
    return path


def list_images(paths) -> list[str]:
    out = []
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                fp = os.path.join(p, name)
                if os.path.isfile(fp) and name.lower().endswith(SUPPORTED_EXT):
                    out.append(fp)
        elif os.path.isfile(p) and p.lower().endswith(SUPPORTED_EXT):
            out.append(p)
    return out
