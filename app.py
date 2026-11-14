"""DustHeal — удаление пыли, пятен и грязи с фотографий.

Запуск интерфейса:   DustHeal.exe [снимок]
Пакет из консоли:    DustHeal.exe --batch карта.json --out папка [--refine] [--skip-invisible]
                                  [--autodetect] [--format .tif] файлы_или_папки...
"""
import argparse
import os
import sys


def _attach_console():
    """Оконный exe не имеет консоли — подключаемся к консоли, из которой запущен (cmd/PowerShell)."""
    if os.name == "nt" and (sys.stdout is None or not hasattr(sys.stdout, "fileno")):
        try:
            import ctypes
            if ctypes.windll.kernel32.AttachConsole(-1):
                ctypes.windll.kernel32.SetConsoleOutputCP(65001)
                sys.stdout = open("CONOUT$", "w", encoding="utf-8", buffering=1)
                sys.stderr = sys.stdout
        except Exception:
            pass
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = sys.stdout
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _cli(argv):
    _attach_console()
    from dustheal.core.batch import BatchOptions, run_batch
    from dustheal.core.imageio import list_images
    from dustheal.core.spots import load_template

    p = argparse.ArgumentParser(prog="DustHeal", description="Пакетное удаление пыли")
    p.add_argument("--batch", metavar="MAP.json", help="карта пятен (пусто + --autodetect = только автопоиск)")
    p.add_argument("--out", default="", help="папка для результатов (по умолчанию ./cleaned рядом со снимком)")
    p.add_argument("--suffix", default="_clean")
    p.add_argument("--format", default="same", choices=["same", ".jpg", ".tif", ".png"])
    p.add_argument("--quality", type=int, default=97)
    p.add_argument("--no-refine", action="store_true", help="не уточнять положение пятен")
    p.add_argument("--skip-invisible", action="store_true")
    p.add_argument("--autodetect", action="store_true")
    p.add_argument("--sensitivity", type=float, default=0.5)
    p.add_argument("--method", default="", choices=["", "auto", "dust", "heal", "fill"])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("inputs", nargs="+")
    a = p.parse_args(argv)
    tpl = load_template(a.batch) if a.batch else None
    files = list_images(a.inputs)
    opt = BatchOptions(out_dir=a.out, suffix=a.suffix, out_format=a.format, jpeg_quality=a.quality,
                       refine=not a.no_refine, skip_invisible=a.skip_invisible, autodetect=a.autodetect,
                       sensitivity=a.sensitivity, force_method=a.method, workers=a.workers,
                       overwrite=a.overwrite)
    res = run_batch(files, tpl, opt, lambda d, t, path, ok, msg: print(f"[{d}/{t}] {'OK' if ok else 'ERR'} {path}: {msg}", flush=True))
    return 0 if all(r[1] for r in res) else 1


def main():
    if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        sys.exit(_cli(sys.argv[1:]))
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QPalette, QColor
    from dustheal.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("DustHeal")
    app.setStyle("Fusion")
    pal = QPalette()
    for role, c in ((QPalette.Window, (45, 45, 48)), (QPalette.WindowText, (225, 225, 225)),
                    (QPalette.Base, (30, 30, 32)), (QPalette.AlternateBase, (45, 45, 48)),
                    (QPalette.Text, (225, 225, 225)), (QPalette.Button, (55, 55, 60)),
                    (QPalette.ButtonText, (225, 225, 225)), (QPalette.Highlight, (60, 120, 200)),
                    (QPalette.HighlightedText, (255, 255, 255)), (QPalette.ToolTipBase, (60, 60, 60)),
                    (QPalette.ToolTipText, (230, 230, 230)), (QPalette.PlaceholderText, (130, 130, 130))):
        pal.setColor(role, QColor(*c))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(120, 120, 120))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(120, 120, 120))
    pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor(120, 120, 120))
    app.setPalette(pal)
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        w.open_file(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
