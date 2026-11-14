"""Главное окно редактора."""
from __future__ import annotations

import os

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QActionGroup, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QMainWindow, QMenu, QMessageBox, QProgressDialog, QPushButton,
                               QSlider, QSpinBox, QToolBar, QVBoxLayout, QWidget)

from ..core.detect import detect_spots
from ..core.heal import heal_all, heal_spot, spot_rect
from ..core.imageio import SUPPORTED_EXT, LoadedImage, load_image, save_image
from ..core.spots import (METHODS, Spot, load_template, save_template, spots_to_template,
                          template_to_spots)
from .batch_dialog import BatchDialog
from .canvas import ImageCanvas, array_to_qimage

APP_NAME = "DustHeal"
IMG_FILTER = "Изображения (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp);;Все файлы (*.*)"


def _rects_intersect(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


class Document:
    def __init__(self, img: LoadedImage):
        self.img = img
        self.result = img.data.copy()
        self.spots: list[Spot] = []
        self.margin = 0.25
        self.dirty = False

    def render_all(self, progress=None):
        self.result = self.img.data.copy()
        heal_all(self.result, self.spots, self.margin, progress)

    def add_spot(self, spot: Spot) -> bool:
        """Возвращает True, если понадобилась полная перерисовка."""
        rect = spot_rect(spot, self.margin)
        need_full = any(s.src_rect and _rects_intersect(rect, s.src_rect) for s in self.spots)
        self.spots.append(spot)
        if need_full:
            self.render_all()
        else:
            heal_spot(self.result, spot, (), self.margin)
        self.dirty = True
        return need_full


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1400, 900)
        self.settings = QSettings(APP_NAME, APP_NAME)
        self.doc: Document | None = None
        self.undo_stack: list = []
        self.redo_stack: list = []
        self.orig_pixmap: QPixmap | None = None
        self.showing_before = False
        icon_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.canvas = ImageCanvas(self)
        self.setCentralWidget(self.canvas)
        self.canvas.spotRequested.connect(self.on_spot_requested)
        self.canvas.contextRequested.connect(self.on_context)
        self.canvas.cursorMoved.connect(self.on_cursor)
        self.canvas.zoomChanged.connect(lambda z: self.lbl_zoom.setText(f"{z * 100:.0f}%"))
        self.canvas.radiusChanged.connect(self._sync_radius)
        self.setAcceptDrops(True)

        self._build_actions()
        self._build_panel()
        self._build_status()
        self.canvas.set_radius(float(self.settings.value("radius", 15.0)))
        self.set_tool("spot")
        self._update_ui()

    # ================================================================ UI
    def _act(self, text, slot, shortcut=None, checkable=False, tip=None):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.setCheckable(checkable)
        if tip:
            a.setToolTip(tip)
            a.setStatusTip(tip)
        a.triggered.connect(slot)
        self.addAction(a)
        return a

    def _build_actions(self):
        self.a_open = self._act("Открыть…", self.open_dialog, "Ctrl+O")
        self.a_save = self._act("Сохранить как…", self.save_dialog, "Ctrl+S")
        self.a_undo = self._act("Отменить", self.undo, "Ctrl+Z")
        self.a_redo = self._act("Повторить", self.redo, "Ctrl+Y")
        self.a_fit = self._act("Вписать", lambda: self.canvas.fit(), "Ctrl+0")
        self.a_100 = self._act("100%", lambda: self.canvas.set_zoom(1.0), "Ctrl+1")
        self.a_before = self._act("До / После", self.toggle_before, "\\", checkable=True,
                                  tip="Показать исходник (клавиша \\)")
        self.a_markers = self._act("Маркеры", self.toggle_markers, "M", checkable=True,
                                   tip="Показать/скрыть контуры пятен (M)")
        self.a_markers.setChecked(True)
        self.a_batch = self._act("Пакетная обработка…", self.open_batch, "Ctrl+B")

        grp = QActionGroup(self)
        self.a_tool_spot = self._act("Точка", lambda: self.set_tool("spot"), "S", True,
                                     "Клик по пятну — удалить (S). Размер: [ ] или Shift+колесо")
        self.a_tool_brush = self._act("Кисть", lambda: self.set_tool("brush"), "B", True,
                                      "Мазок для волосков, царапин, вытянутых пятен (B)")
        self.a_tool_pan = self._act("Рука", lambda: self.set_tool("pan"), "H", True,
                                    "Перемещение (H). Также: пробел или средняя кнопка мыши")
        for a in (self.a_tool_spot, self.a_tool_brush, self.a_tool_pan):
            grp.addAction(a)
        self._act("Меньше", lambda: self.canvas.set_radius(self.canvas.radius / 1.15), "[")
        self._act("Больше", lambda: self.canvas.set_radius(self.canvas.radius * 1.15), "]")

        tb = QToolBar("Основное", self)
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(tb)
        for a in (self.a_open, self.a_save, None, self.a_undo, self.a_redo, None,
                  self.a_tool_spot, self.a_tool_brush, self.a_tool_pan, None,
                  self.a_fit, self.a_100, self.a_before, self.a_markers, None, self.a_batch):
            tb.addSeparator() if a is None else tb.addAction(a)

        m = self.menuBar().addMenu("Файл")
        m.addAction(self.a_open)
        m.addAction(self.a_save)
        m.addSeparator()
        m.addAction(self._act("Сохранить карту пятен…", self.save_map))
        m.addAction(self._act("Загрузить карту пятен…", self.load_map))
        m.addSeparator()
        m.addAction(self.a_batch)
        m.addSeparator()
        m.addAction(self._act("Выход", self.close, "Ctrl+Q"))
        m = self.menuBar().addMenu("Правка")
        m.addAction(self.a_undo)
        m.addAction(self.a_redo)
        m.addAction(self._act("Удалить все пятна", self.clear_spots))
        m.addAction(self._act("Пересчитать всё", self.rerender))
        m = self.menuBar().addMenu("Справка")
        m.addAction(self._act("Как пользоваться", self.show_help, "F1"))

    def _build_panel(self):
        dock = QDockWidget("Инструменты", self)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        w = QWidget()
        lay = QVBoxLayout(w)

        g = QGroupBox("Кисть восстановления")
        f = QFormLayout(g)
        self.sl_radius = QSlider(Qt.Horizontal)
        self.sl_radius.setRange(1, 400)
        self.sp_radius = QSpinBox()
        self.sp_radius.setRange(1, 1000)
        self.sp_radius.setSuffix(" px")
        rl = QHBoxLayout()
        rl.addWidget(self.sl_radius)
        rl.addWidget(self.sp_radius)
        self.sl_radius.valueChanged.connect(lambda v: self.canvas.set_radius(float(v)))
        self.sp_radius.valueChanged.connect(lambda v: self.canvas.set_radius(float(v)))
        f.addRow("Радиус:", rl)
        self.cb_method = QComboBox()
        for k, v in METHODS.items():
            self.cb_method.addItem(v, k)
        self.cb_method.setCurrentIndex(max(0, self.cb_method.findData(self.settings.value("method", "auto"))))
        f.addRow("Метод:", self.cb_method)
        self.sp_margin = QSpinBox()
        self.sp_margin.setRange(5, 100)
        self.sp_margin.setValue(int(self.settings.value("margin", 25)))
        self.sp_margin.setSuffix(" %")
        self.sp_margin.setToolTip("Запас вокруг пятна для мягкого ореола и плавного перехода")
        self.sp_margin.valueChanged.connect(self._margin_changed)
        f.addRow("Растушёвка/запас:", self.sp_margin)
        b = QPushButton("Применить метод ко всем пятнам")
        b.clicked.connect(self.apply_method_all)
        f.addRow(b)
        lay.addWidget(g)

        g = QGroupBox("Автопоиск пятен")
        f = QFormLayout(g)
        self.sl_sens = QSlider(Qt.Horizontal)
        self.sl_sens.setRange(0, 100)
        self.sl_sens.setValue(int(self.settings.value("sens", 50)))
        self.lbl_sens = QLabel()
        self.sl_sens.valueChanged.connect(lambda v: self.lbl_sens.setText(f"{v}"))
        self.lbl_sens.setText(str(self.sl_sens.value()))
        sl = QHBoxLayout()
        sl.addWidget(self.sl_sens)
        sl.addWidget(self.lbl_sens)
        f.addRow("Чувствительность:", sl)
        self.sp_maxsize = QDoubleSpinBox()
        self.sp_maxsize.setRange(0.2, 10.0)
        self.sp_maxsize.setSingleStep(0.2)
        self.sp_maxsize.setValue(float(self.settings.value("maxsize", 2.0)))
        self.sp_maxsize.setSuffix(" % кадра")
        f.addRow("Макс. радиус:", self.sp_maxsize)
        self.sp_minsize = QSpinBox()
        self.sp_minsize.setRange(1, 100)
        self.sp_minsize.setValue(int(self.settings.value("minsize", 2)))
        self.sp_minsize.setSuffix(" px")
        f.addRow("Мин. радиус:", self.sp_minsize)
        self.ch_smooth = QCheckBox("Только на однородном фоне")
        self.ch_smooth.setChecked(True)
        self.ch_smooth.setToolTip("Небо, вода, стены, студийный фон. Снижает ложные срабатывания на деталях сюжета")
        f.addRow(self.ch_smooth)
        self.ch_light = QCheckBox("Искать и светлые пятна")
        f.addRow(self.ch_light)
        b = QPushButton("Найти пятна")
        b.clicked.connect(self.autodetect)
        f.addRow(b)
        lay.addWidget(g)

        g = QGroupBox("Карта пятен (для пакета)")
        v = QVBoxLayout(g)
        lbl = QLabel("Пыль на матрице остаётся на месте от кадра к кадру. "
                     "Отметьте пятна на одном снимке и сохраните карту или сразу запустите пакет.")
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color: gray;")
        v.addWidget(lbl)
        hb = QHBoxLayout()
        b1 = QPushButton("Сохранить…")
        b1.clicked.connect(self.save_map)
        b2 = QPushButton("Загрузить…")
        b2.clicked.connect(self.load_map)
        hb.addWidget(b1)
        hb.addWidget(b2)
        v.addLayout(hb)
        b = QPushButton("Пакетная обработка…")
        b.clicked.connect(self.open_batch)
        v.addWidget(b)
        lay.addWidget(g)

        self.lbl_info = QLabel()
        self.lbl_info.setWordWrap(True)
        lay.addWidget(self.lbl_info)
        legend = QLabel("<small>Цвет маркера — применённый метод:<br>"
                        "<span style='color:#50dcff'>■</span> пыль (снятие затемнения)<br>"
                        "<span style='color:#ffc83c'>■</span> восстановление (текстура)<br>"
                        "<span style='color:#8cff78'>■</span> плавная заливка<br><br>"
                        "ЛКМ — удалить пятно · ПКМ — меню пятна<br>"
                        "Колесо — масштаб · Пробел/СКМ — сдвиг<br>"
                        "[ ] или Shift+колесо — размер · \\ — до/после</small>")
        legend.setWordWrap(True)
        lay.addWidget(legend)
        lay.addStretch(1)
        dock.setWidget(w)
        dock.setMinimumWidth(300)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _build_status(self):
        sb = self.statusBar()
        self.lbl_pos = QLabel()
        self.lbl_zoom = QLabel()
        self.lbl_count = QLabel()
        sb.addPermanentWidget(self.lbl_count)
        sb.addPermanentWidget(self.lbl_pos)
        sb.addPermanentWidget(self.lbl_zoom)

    def _sync_radius(self, r):
        for wdg in (self.sl_radius, self.sp_radius):
            wdg.blockSignals(True)
            wdg.setValue(int(round(r)))
            wdg.blockSignals(False)

    def _margin_changed(self, v):
        if self.doc:
            self.doc.margin = v / 100.0

    def set_tool(self, t):
        self.canvas.set_tool(t)
        {"spot": self.a_tool_spot, "brush": self.a_tool_brush, "pan": self.a_tool_pan}[t].setChecked(True)

    def _update_ui(self):
        has = self.doc is not None
        for a in (self.a_save, self.a_fit, self.a_100, self.a_before):
            a.setEnabled(has)
        self.a_undo.setEnabled(bool(self.undo_stack))
        self.a_redo.setEnabled(bool(self.redo_stack))
        if has:
            n = len(self.doc.spots)
            self.lbl_count.setText(f"Пятен: {n}")
            im = self.doc.img
            name = os.path.basename(im.path)
            self.setWindowTitle(f"{name}{' *' if self.doc.dirty else ''} — {APP_NAME}")
            self.lbl_info.setText(f"<small>{name}<br>{im.width}×{im.height}, {im.bit_depth} бит/канал"
                                  f"{', ICC' if im.icc else ''}{', EXIF' if im.exif else ''}</small>")
        else:
            self.lbl_count.setText("")
            self.lbl_info.setText("<small>Откройте снимок (Ctrl+O) или перетащите файл в окно.</small>")

    # ================================================================ файлы
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = [u.toLocalFile() for u in e.mimeData().urls()]
        imgs = [u for u in urls if u.lower().endswith(SUPPORTED_EXT)]
        maps = [u for u in urls if u.lower().endswith(".json")]
        if imgs:
            self.open_file(imgs[0])
        if maps and self.doc:
            self._apply_map_file(maps[0])

    def _maybe_save(self) -> bool:
        if self.doc and self.doc.dirty and self.doc.spots:
            r = QMessageBox.question(self, APP_NAME, "Результат не сохранён. Продолжить без сохранения?",
                                     QMessageBox.Yes | QMessageBox.No)
            return r == QMessageBox.Yes
        return True

    def open_dialog(self):
        if not self._maybe_save():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Открыть снимок", self.settings.value("dir", ""), IMG_FILTER)
        if path:
            self.open_file(path)

    def open_file(self, path):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            img = load_image(path)
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, APP_NAME, str(e))
            return
        prev_spots = self.doc.spots if self.doc else []
        prev_size = (self.doc.img.width, self.doc.img.height) if self.doc else None
        self.doc = Document(img)
        self.doc.margin = self.sp_margin.value() / 100.0
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.settings.setValue("dir", os.path.dirname(path))
        self.orig_pixmap = QPixmap.fromImage(array_to_qimage(img.data))
        self.canvas.set_image(img.data, fit=True)
        self.canvas.set_spots([])
        self.a_before.setChecked(False)
        self.showing_before = False
        QApplication.restoreOverrideCursor()
        self._update_ui()
        if prev_spots:
            r = QMessageBox.question(self, APP_NAME,
                                     f"Перенести {len(prev_spots)} пятен с предыдущего снимка на этот?\n"
                                     "(пыль на матрице обычно в тех же местах)",
                                     QMessageBox.Yes | QMessageBox.No)
            if r == QMessageBox.Yes:
                tpl = spots_to_template(prev_spots, *prev_size)
                self._set_spots(template_to_spots(tpl, img.width, img.height), push=True)

    def save_dialog(self):
        if not self.doc:
            return
        src = self.doc.img.path
        base, ext = os.path.splitext(src)
        default = base + "_clean" + (ext if ext.lower() != ".bmp" else ".png")
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результат", default,
            "Как исходник (*%s);;JPEG (*.jpg);;TIFF (*.tif);;PNG (*.png)" % ext)
        if not path:
            return
        if os.path.abspath(path) == os.path.abspath(src):
            if QMessageBox.question(self, APP_NAME, "Перезаписать исходный файл?",
                                    QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return
        if not os.path.splitext(path)[1]:
            path += ext
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            save_image(path, self.doc.img, self.doc.result, int(self.settings.value("jpegq", 97)))
            self.doc.dirty = False
            self.statusBar().showMessage(f"Сохранено: {path}", 6000)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, APP_NAME, f"Ошибка сохранения: {e}")
        finally:
            QApplication.restoreOverrideCursor()
        self._update_ui()

    def save_map(self):
        if not self.doc or not self.doc.spots:
            QMessageBox.information(self, APP_NAME, "Нет отмеченных пятен.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить карту пятен",
                                              os.path.join(self.settings.value("dir", ""), "dust_map.json"),
                                              "Карта пятен (*.json)")
        if path:
            save_template(path, spots_to_template(self.doc.spots, self.doc.img.width, self.doc.img.height))
            self.statusBar().showMessage(f"Карта сохранена: {path}", 6000)

    def load_map(self):
        if not self.doc:
            QMessageBox.information(self, APP_NAME, "Сначала откройте снимок.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить карту пятен", self.settings.value("dir", ""),
                                              "Карта пятен (*.json)")
        if path:
            self._apply_map_file(path)

    def _apply_map_file(self, path):
        try:
            tpl = load_template(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, APP_NAME, str(e))
            return
        spots = template_to_spots(tpl, self.doc.img.width, self.doc.img.height)
        if self.doc.spots:
            r = QMessageBox.question(self, APP_NAME, "Добавить к существующим пятнам? (Нет — заменить)",
                                     QMessageBox.Yes | QMessageBox.No)
            if r == QMessageBox.Yes:
                spots = [s.copy() for s in self.doc.spots] + spots
        self._set_spots(spots, push=True)

    # ================================================================ правка
    def _snapshot(self):
        return [s.copy() for s in self.doc.spots]

    def _push_undo(self):
        self.undo_stack.append(self._snapshot())
        if len(self.undo_stack) > 200:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def _set_spots(self, spots, push=True):
        if push:
            self._push_undo()
        self.doc.spots = spots
        self.rerender()

    def rerender(self):
        if not self.doc:
            return
        n = len(self.doc.spots)
        dlg = None
        if n > 15:
            dlg = QProgressDialog("Обработка пятен…", None, 0, n, self)
            dlg.setWindowModality(Qt.WindowModal)
            dlg.setMinimumDuration(300)

        def prog(i, total):
            if dlg:
                dlg.setValue(i)
                QApplication.processEvents()

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.doc.render_all(prog)
        finally:
            QApplication.restoreOverrideCursor()
            if dlg:
                dlg.close()
        self.doc.dirty = True
        self._refresh_view()

    def _refresh_view(self):
        if not self.showing_before:
            self.canvas.set_pixmap(QPixmap.fromImage(array_to_qimage(self.doc.result)))
        self.canvas.set_spots(self.doc.spots)
        self._update_ui()

    def on_spot_requested(self, points, r):
        if not self.doc:
            return
        if self.showing_before:
            self.a_before.setChecked(False)
            self.toggle_before()
        self._push_undo()
        spot = Spot([tuple(p) for p in points], float(r), self.cb_method.currentData())
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.doc.add_spot(spot)
        finally:
            QApplication.restoreOverrideCursor()
        self._refresh_view()
        self.statusBar().showMessage(f"Метод: {METHODS.get(spot.used_method, spot.used_method)}", 3000)

    def _spot_at(self, x, y):
        tol = 3.0 / max(self.canvas.zoom(), 1e-3)
        for i in range(len(self.doc.spots) - 1, -1, -1):
            if self.doc.spots[i].contains(x, y, tol):
                return i
        return -1

    def on_context(self, x, y, gpos):
        if not self.doc:
            return
        i = self._spot_at(x, y)
        menu = QMenu(self)
        if i >= 0:
            s = self.doc.spots[i]
            menu.addAction(f"Применён: {METHODS.get(s.used_method, s.used_method)}").setEnabled(False)
            menu.addSeparator()
            menu.addAction("Удалить пятно", lambda: self.remove_spot(i))
            sub = menu.addMenu("Метод для этого пятна")
            for k, v in METHODS.items():
                a = sub.addAction(v, lambda k=k: self.set_spot_method(i, k))
                a.setCheckable(True)
                a.setChecked(s.method == k)
            menu.addAction("Увеличить на 20%", lambda: self.scale_spot(i, 1.2))
            menu.addAction("Уменьшить на 20%", lambda: self.scale_spot(i, 1 / 1.2))
        else:
            menu.addAction("Здесь нет пятна").setEnabled(False)
        menu.addSeparator()
        menu.addAction("Удалить все пятна", self.clear_spots)
        menu.exec(gpos)

    def remove_spot(self, i):
        sp = self._snapshot()
        del sp[i]
        self._set_spots(sp)

    def set_spot_method(self, i, m):
        sp = self._snapshot()
        sp[i].method = m
        self._set_spots(sp)

    def scale_spot(self, i, f):
        sp = self._snapshot()
        sp[i].r *= f
        self._set_spots(sp)

    def clear_spots(self):
        if self.doc and self.doc.spots:
            self._set_spots([])

    def apply_method_all(self):
        if not self.doc or not self.doc.spots:
            return
        m = self.cb_method.currentData()
        sp = self._snapshot()
        for s in sp:
            s.method = m
        self._set_spots(sp)

    def undo(self):
        if not self.doc or not self.undo_stack:
            return
        self.redo_stack.append(self._snapshot())
        self.doc.spots = self.undo_stack.pop()
        self.rerender()

    def redo(self):
        if not self.doc or not self.redo_stack:
            return
        self.undo_stack.append(self._snapshot())
        self.doc.spots = self.redo_stack.pop()
        self.rerender()

    def autodetect(self):
        if not self.doc:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            im = self.doc.img
            found = detect_spots(im.data, self.sl_sens.value() / 100.0, self.sp_minsize.value(),
                                 max(im.width, im.height) * self.sp_maxsize.value() / 100.0,
                                 self.ch_smooth.isChecked(), not self.ch_light.isChecked())
        finally:
            QApplication.restoreOverrideCursor()
        new = []
        for f in found:
            fx, fy = f.center
            if all(not s.contains(fx, fy, f.r * 0.5) for s in self.doc.spots):
                f.method = self.cb_method.currentData()
                new.append(f)
        self.settings.setValue("sens", self.sl_sens.value())
        self.settings.setValue("maxsize", self.sp_maxsize.value())
        self.settings.setValue("minsize", self.sp_minsize.value())
        if not new:
            QMessageBox.information(self, APP_NAME, "Новых пятен не найдено. Попробуйте увеличить чувствительность.")
            return
        self._set_spots(self._snapshot() + new)
        self.statusBar().showMessage(f"Найдено пятен: {len(new)}. Лишние удалите через ПКМ.", 8000)

    # ================================================================ вид
    def toggle_before(self):
        if not self.doc:
            return
        self.showing_before = self.a_before.isChecked()
        if self.showing_before:
            self.canvas.set_pixmap(self.orig_pixmap)
            self.statusBar().showMessage("Показан ИСХОДНИК", 0)
        else:
            self.canvas.set_pixmap(QPixmap.fromImage(array_to_qimage(self.doc.result)))
            self.statusBar().clearMessage()

    def toggle_markers(self):
        self.canvas.set_markers_visible(self.a_markers.isChecked())

    def on_cursor(self, x, y):
        if self.doc:
            self.lbl_pos.setText(f"x {int(x)}  y {int(y)}")

    # ================================================================ пакет
    def open_batch(self):
        tpl = None
        if self.doc and self.doc.spots:
            tpl = spots_to_template(self.doc.spots, self.doc.img.width, self.doc.img.height)
        dlg = BatchDialog(self, tpl, self.settings, self.sp_margin.value() / 100.0)
        dlg.exec()

    def show_help(self):
        QMessageBox.information(self, "Как пользоваться", (
            "1. Откройте снимок (лучше с пылью на ровном фоне — небо, стена).\n"
            "2. Кликните по пятну инструментом «Точка» — радиус чуть больше пятна.\n"
            "   Для волосков и царапин используйте «Кисть».\n"
            "   Или нажмите «Найти пятна» — проверьте результат, лишнее удалите (ПКМ).\n"
            "3. Метод «Авто» сам выбирает:\n"
            "   • Пыль — снимает тень пятна, сохраняя детали и зерно под ним;\n"
            "   • Восстановление — переносит текстуру с соседнего участка и\n"
            "     подгоняет яркость/цвет по Пуассону;\n"
            "   • Плавная заливка — идеально гладкий градиент + родное зерно.\n"
            "4. \\ — сравнить до/после. Ctrl+S — сохранить.\n"
            "5. Пакет: отметьте пятна на одном кадре → «Пакетная обработка» →\n"
            "   добавьте файлы. Положение каждого пятна уточняется на каждом кадре,\n"
            "   а цвет, яркость и контраст фона берутся из окружения на этом кадре.\n\n"
            "Снимки показываются в ориентации матрицы (без поворота по EXIF) —\n"
            "так пыль на всех кадрах в одних и тех же координатах."))

    def closeEvent(self, e):
        if not self._maybe_save():
            e.ignore()
            return
        self.settings.setValue("radius", self.canvas.radius)
        self.settings.setValue("method", self.cb_method.currentData())
        self.settings.setValue("margin", self.sp_margin.value())
        e.accept()
