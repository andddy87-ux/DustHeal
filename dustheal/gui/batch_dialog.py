"""Окно пакетной обработки."""
from __future__ import annotations

import os
import threading

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                               QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                               QSpinBox, QVBoxLayout)

from ..core.batch import BatchOptions, run_batch
from ..core.imageio import list_images
from ..core.spots import METHODS, load_template

IMG_FILTER = "Изображения (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp)"


class _Worker(QObject):
    progress = Signal(int, int, str, bool, str)
    finished = Signal(list)

    def __init__(self, files, tpl, opt):
        super().__init__()
        self.files, self.tpl, self.opt = files, tpl, opt
        self.stop = threading.Event()

    def run(self):
        res = run_batch(self.files, self.tpl, self.opt,
                        lambda *a: self.progress.emit(*a), self.stop.is_set)
        self.finished.emit(res)


class BatchDialog(QDialog):
    def __init__(self, parent, template, settings, margin):
        super().__init__(parent)
        self.setWindowTitle("Пакетная обработка")
        self.resize(820, 760)
        self.settings = settings
        self.template = template
        self.margin = margin
        self.thread = None
        self.worker = None
        lay = QVBoxLayout(self)

        g = QGroupBox("1. Карта пятен")
        v = QVBoxLayout(g)
        self.lbl_tpl = QLabel()
        self.lbl_tpl.setWordWrap(True)
        v.addWidget(self.lbl_tpl)
        hb = QHBoxLayout()
        b = QPushButton("Загрузить карту из файла…")
        b.clicked.connect(self.load_tpl)
        hb.addWidget(b)
        self.ch_noTpl = QCheckBox("Без карты — только автопоиск на каждом снимке")
        self.ch_noTpl.toggled.connect(self._upd_tpl)
        hb.addWidget(self.ch_noTpl)
        hb.addStretch(1)
        v.addLayout(hb)
        lay.addWidget(g)

        g = QGroupBox("2. Снимки")
        v = QVBoxLayout(g)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.setMinimumHeight(140)
        v.addWidget(self.list)
        hb = QHBoxLayout()
        for text, slot in (("Добавить файлы…", self.add_files), ("Добавить папку…", self.add_folder),
                           ("Убрать выбранные", self.remove_sel), ("Очистить", self.list.clear)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            hb.addWidget(b)
        hb.addStretch(1)
        v.addLayout(hb)
        lay.addWidget(g)

        g = QGroupBox("3. Адаптация к каждому снимку")
        f = QFormLayout(g)
        self.ch_refine = QCheckBox("Уточнять положение и размер каждого пятна (пыль слегка «плывёт» "
                                   "с диафрагмой и фокусным)")
        self.ch_refine.setChecked(settings.value("b_refine", "true") == "true")
        f.addRow(self.ch_refine)
        self.sp_search = QSpinBox()
        self.sp_search.setRange(10, 300)
        self.sp_search.setValue(int(settings.value("b_search", 60)))
        self.sp_search.setSuffix(" % радиуса")
        f.addRow("Зона поиска смещения:", self.sp_search)
        self.ch_skip = QCheckBox("Не трогать пятна, которых на данном снимке не видно")
        self.ch_skip.setToolTip("Например, на открытой диафрагме или на пёстром фоне пыль не видна — "
                                "такие места останутся нетронутыми")
        self.ch_skip.setChecked(settings.value("b_skip", "false") == "true")
        f.addRow(self.ch_skip)
        hb = QHBoxLayout()
        self.ch_auto = QCheckBox("Дополнительно искать новые пятна, чувствительность:")
        self.ch_auto.setChecked(settings.value("b_auto", "false") == "true")
        self.sp_sens = QSpinBox()
        self.sp_sens.setRange(0, 100)
        self.sp_sens.setValue(int(settings.value("sens", 50)))
        hb.addWidget(self.ch_auto)
        hb.addWidget(self.sp_sens)
        hb.addStretch(1)
        f.addRow(hb)
        self.cb_rot = QComboBox()
        for k, t in (("auto_cw", "Автоматически, кадр повёрнут по часовой"),
                     ("auto_ccw", "Автоматически, кадр повёрнут против часовой"),
                     ("none", "Не поворачивать карту")):
            self.cb_rot.addItem(t, k)
        self.cb_rot.setCurrentIndex(max(0, self.cb_rot.findData(settings.value("b_rot", "auto_cw"))))
        self.cb_rot.setToolTip("Если у снимков другая ориентация (уже повёрнутые портретные кадры), "
                               "карта пятен поворачивается вместе с кадром")
        f.addRow("Другая ориентация:", self.cb_rot)
        self.cb_method = QComboBox()
        self.cb_method.addItem("Как в карте", "")
        for k, t in METHODS.items():
            self.cb_method.addItem(t, k)
        f.addRow("Метод:", self.cb_method)
        lay.addWidget(g)

        g = QGroupBox("4. Сохранение")
        f = QFormLayout(g)
        hb = QHBoxLayout()
        self.ed_out = QLineEdit(settings.value("b_out", ""))
        self.ed_out.setPlaceholderText("пусто = подпапка «cleaned» рядом с каждым снимком")
        b = QPushButton("…")
        b.setFixedWidth(32)
        b.clicked.connect(self.pick_out)
        hb.addWidget(self.ed_out)
        hb.addWidget(b)
        f.addRow("Папка:", hb)
        self.ed_suffix = QLineEdit(settings.value("b_suffix", "_clean"))
        f.addRow("Суффикс имени:", self.ed_suffix)
        hb = QHBoxLayout()
        self.cb_fmt = QComboBox()
        for k, t in (("same", "Как исходник"), (".jpg", "JPEG"), (".tif", "TIFF (16 бит сохраняется)"),
                     (".png", "PNG")):
            self.cb_fmt.addItem(t, k)
        self.cb_fmt.setCurrentIndex(max(0, self.cb_fmt.findData(settings.value("b_fmt", "same"))))
        self.sp_q = QSpinBox()
        self.sp_q.setRange(50, 100)
        self.sp_q.setValue(int(settings.value("jpegq", 97)))
        hb.addWidget(self.cb_fmt)
        hb.addWidget(QLabel("Качество JPEG:"))
        hb.addWidget(self.sp_q)
        hb.addStretch(1)
        f.addRow("Формат:", hb)
        hb = QHBoxLayout()
        self.ch_meta = QCheckBox("Сохранять EXIF и цветовой профиль")
        self.ch_meta.setChecked(True)
        self.ch_over = QCheckBox("Перезаписывать существующие результаты")
        self.sp_workers = QSpinBox()
        self.sp_workers.setRange(1, max(1, os.cpu_count() or 2))
        self.sp_workers.setValue(int(settings.value("b_workers", min(4, max(1, (os.cpu_count() or 2) // 2)))))
        hb.addWidget(self.ch_meta)
        hb.addWidget(self.ch_over)
        hb.addWidget(QLabel("Потоков:"))
        hb.addWidget(self.sp_workers)
        hb.addStretch(1)
        f.addRow(hb)
        lay.addWidget(g)

        self.bar = QProgressBar()
        lay.addWidget(self.bar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        lay.addWidget(self.log, 1)
        hb = QHBoxLayout()
        hb.addStretch(1)
        self.b_run = QPushButton("Запустить")
        self.b_run.setDefault(True)
        self.b_run.clicked.connect(self.run)
        self.b_stop = QPushButton("Остановить")
        self.b_stop.setEnabled(False)
        self.b_stop.clicked.connect(self.stop)
        self.b_close = QPushButton("Закрыть")
        self.b_close.clicked.connect(self.close)
        for b in (self.b_run, self.b_stop, self.b_close):
            hb.addWidget(b)
        lay.addLayout(hb)
        self.setAcceptDrops(True)
        self._upd_tpl()

    # ------------------------------------------------------------
    def _upd_tpl(self):
        if self.ch_noTpl.isChecked():
            self.lbl_tpl.setText("Карта не используется: на каждом снимке пятна ищутся автоматически.")
            self.ch_auto.setChecked(True)
        elif self.template:
            n = len(self.template["spots"])
            self.lbl_tpl.setText(f"Используется карта: <b>{n}</b> пятен (снимок-образец "
                                 f"{self.template['ref_width']}×{self.template['ref_height']}).")
        else:
            self.lbl_tpl.setText("<span style='color:#c33'>Карта пятен не задана.</span> Отметьте пятна "
                                 "в редакторе или загрузите карту из файла.")

    def load_tpl(self):
        p, _ = QFileDialog.getOpenFileName(self, "Карта пятен", self.settings.value("dir", ""),
                                           "Карта пятен (*.json)")
        if p:
            try:
                self.template = load_template(p)
                self.ch_noTpl.setChecked(False)
            except Exception as e:  # noqa: BLE001
                QMessageBox.critical(self, "Ошибка", str(e))
            self._upd_tpl()

    def _add(self, paths):
        existing = {self.list.item(i).text() for i in range(self.list.count())}
        for p in list_images(paths):
            p = os.path.normpath(p)
            if p not in existing and os.sep + "cleaned" + os.sep not in p:
                self.list.addItem(p)
                existing.add(p)

    def add_files(self):
        ps, _ = QFileDialog.getOpenFileNames(self, "Снимки", self.settings.value("dir", ""), IMG_FILTER)
        self._add(ps)

    def add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Папка со снимками", self.settings.value("dir", ""))
        if d:
            self._add([d])

    def remove_sel(self):
        for it in self.list.selectedItems():
            self.list.takeItem(self.list.row(it))

    def pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "Папка для результатов", self.ed_out.text())
        if d:
            self.ed_out.setText(d)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self._add([u.toLocalFile() for u in e.mimeData().urls()])

    # ------------------------------------------------------------
    def options(self) -> BatchOptions:
        return BatchOptions(
            out_dir=self.ed_out.text().strip(), suffix=self.ed_suffix.text(),
            out_format=self.cb_fmt.currentData(), jpeg_quality=self.sp_q.value(),
            keep_metadata=self.ch_meta.isChecked(), overwrite=self.ch_over.isChecked(),
            refine=self.ch_refine.isChecked(), search_frac=self.sp_search.value() / 100.0,
            skip_invisible=self.ch_skip.isChecked(), autodetect=self.ch_auto.isChecked(),
            sensitivity=self.sp_sens.value() / 100.0, rotate=self.cb_rot.currentData(),
            force_method=self.cb_method.currentData(), margin_frac=self.margin,
            workers=self.sp_workers.value())

    def run(self):
        files = [self.list.item(i).text() for i in range(self.list.count())]
        if not files:
            QMessageBox.information(self, "Пакет", "Добавьте снимки.")
            return
        tpl = None if self.ch_noTpl.isChecked() else self.template
        if tpl is None and not self.ch_auto.isChecked():
            QMessageBox.information(self, "Пакет", "Нет карты пятен и выключен автопоиск — нечего делать.")
            return
        opt = self.options()
        if not opt.suffix and not opt.out_dir and opt.out_format == "same":
            pass  # пишется в подпапку cleaned — исходники не затрагиваются
        s = self.settings
        s.setValue("b_refine", "true" if opt.refine else "false")
        s.setValue("b_search", self.sp_search.value())
        s.setValue("b_skip", "true" if opt.skip_invisible else "false")
        s.setValue("b_auto", "true" if self.ch_auto.isChecked() else "false")
        s.setValue("b_rot", opt.rotate)
        s.setValue("b_out", opt.out_dir)
        s.setValue("b_suffix", opt.suffix)
        s.setValue("b_fmt", opt.out_format)
        s.setValue("jpegq", opt.jpeg_quality)
        s.setValue("b_workers", opt.workers)

        self.bar.setRange(0, len(files))
        self.bar.setValue(0)
        self.log.appendPlainText(f"Старт: {len(files)} снимков…")
        self.thread = QThread(self)
        self.worker = _Worker(files, tpl, opt)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.b_run.setEnabled(False)
        self.b_stop.setEnabled(True)
        self.thread.start()

    def stop(self):
        if self.worker:
            self.worker.stop.set()
            self.log.appendPlainText("Остановка после текущих снимков…")

    def on_progress(self, done, total, path, ok, msg):
        self.bar.setValue(done)
        self.log.appendPlainText(f"[{done}/{total}] {'OK ' if ok else 'ОШИБКА '}{os.path.basename(path)}: {msg}")

    def on_finished(self, res):
        ok = sum(1 for r in res if r[1])
        self.log.appendPlainText(f"Готово: успешно {ok}, ошибок {len(res) - ok}.")
        self.b_run.setEnabled(True)
        self.b_stop.setEnabled(False)

    def closeEvent(self, e):
        if self.thread and self.thread.isRunning():
            if QMessageBox.question(self, "Пакет", "Обработка идёт. Остановить и закрыть?",
                                    QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                e.ignore()
                return
            self.worker.stop.set()
            self.thread.quit()
            self.thread.wait(60000)
        e.accept()
