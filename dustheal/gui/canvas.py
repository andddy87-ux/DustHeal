"""Холст просмотра: масштаб, панорама, отметка пятен кликом/кистью, маркеры."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsPixmapItem,
                               QGraphicsScene, QGraphicsView)

METHOD_COLORS = {
    "dust": QColor(80, 220, 255),
    "heal": QColor(255, 200, 60),
    "fill": QColor(140, 255, 120),
    "auto": QColor(255, 255, 255),
    "": QColor(255, 255, 255),
}


def array_to_qimage(rgb: np.ndarray) -> QImage:
    if rgb.dtype == np.uint16:
        rgb = (rgb >> 8).astype(np.uint8)
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    qi = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return qi.copy()


class ImageCanvas(QGraphicsView):
    spotRequested = Signal(list, float)      # points, radius
    contextRequested = Signal(float, float, object)  # x, y, global pos
    cursorMoved = Signal(float, float)
    zoomChanged = Signal(float)
    radiusChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QBrush(QColor(38, 38, 40)))
        self.pix = QGraphicsPixmapItem()
        self.pix.setTransformationMode(Qt.SmoothTransformation)
        self.scene().addItem(self.pix)
        self.cursor_item = QGraphicsEllipseItem()
        pen = QPen(QColor(255, 255, 255), 0)
        pen.setCosmetic(True)
        self.cursor_item.setPen(pen)
        self.cursor_item.setZValue(100)
        self.cursor_item.setVisible(False)
        self.scene().addItem(self.cursor_item)
        self.cursor_item2 = QGraphicsEllipseItem()
        pen2 = QPen(QColor(0, 0, 0, 160), 0, Qt.DashLine)
        pen2.setCosmetic(True)
        self.cursor_item2.setPen(pen2)
        self.cursor_item2.setZValue(101)
        self.cursor_item2.setVisible(False)
        self.scene().addItem(self.cursor_item2)
        self.stroke_item = QGraphicsPathItem()
        self.stroke_item.setZValue(90)
        self.scene().addItem(self.stroke_item)
        self.spot_items = []
        self.tool = "spot"          # spot | brush | pan
        self.radius = 15.0
        self.show_markers = True
        self._pan = None
        self._space = False
        self._stroke = None
        self._has_image = False

    # ------------------------------------------------------------ изображение
    def set_image(self, rgb: np.ndarray, fit: bool = False):
        self.pix.setPixmap(QPixmap.fromImage(array_to_qimage(rgb)))
        self.scene().setSceneRect(QRectF(self.pix.boundingRect()).adjusted(-2000, -2000, 2000, 2000))
        self._has_image = True
        if fit:
            self.fit()

    def set_pixmap(self, pm: QPixmap):
        self.pix.setPixmap(pm)

    def fit(self):
        if not self._has_image:
            return
        self.fitInView(self.pix, Qt.KeepAspectRatio)
        self.zoomChanged.emit(self.zoom())

    def zoom(self) -> float:
        return self.transform().m11()

    def set_zoom(self, z: float):
        z = max(0.02, min(32.0, z))
        f = z / self.zoom()
        self.scale(f, f)
        self.zoomChanged.emit(self.zoom())

    # ------------------------------------------------------------ маркеры
    def set_spots(self, spots):
        for it in self.spot_items:
            self.scene().removeItem(it)
        self.spot_items = []
        for s in spots:
            path = QPainterPath()
            if len(s.points) == 1:
                x, y = s.points[0]
                path.addEllipse(QPointF(x, y), s.r, s.r)
            else:
                for x, y in s.points:
                    path.addEllipse(QPointF(x, y), s.r, s.r)
                path = path.simplified()
            item = QGraphicsPathItem(path)
            col = METHOD_COLORS.get(s.used_method or s.method, QColor(255, 255, 255))
            pen = QPen(col, 1.3)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setZValue(50)
            item.setVisible(self.show_markers)
            self.scene().addItem(item)
            self.spot_items.append(item)

    def set_markers_visible(self, v: bool):
        self.show_markers = v
        for it in self.spot_items:
            it.setVisible(v)

    def set_radius(self, r: float):
        self.radius = max(1.0, min(1000.0, r))
        self._update_cursor(self.cursor_item.rect().center() if self.cursor_item.isVisible() else None)
        self.radiusChanged.emit(self.radius)

    def _update_cursor(self, scene_pt):
        if scene_pt is None:
            return
        r = self.radius
        rect = QRectF(scene_pt.x() - r, scene_pt.y() - r, 2 * r, 2 * r)
        self.cursor_item.setRect(rect)
        self.cursor_item2.setRect(rect)
        vis = self._has_image and self.tool in ("spot", "brush") and not self._space
        self.cursor_item.setVisible(vis)
        self.cursor_item2.setVisible(vis)

    # ------------------------------------------------------------ события
    def wheelEvent(self, e):
        if e.modifiers() & (Qt.AltModifier | Qt.ShiftModifier):
            d = e.angleDelta().y() or e.angleDelta().x()
            self.set_radius(self.radius * (1.1 if d > 0 else 1 / 1.1))
            return
        d = e.angleDelta().y()
        if d == 0:
            return
        f = 1.2 ** (d / 120.0)
        z = self.zoom() * f
        if 0.02 <= z <= 32:
            self.scale(f, f)
            self.zoomChanged.emit(self.zoom())

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = True
            self.viewport().setCursor(Qt.OpenHandCursor)
            self.cursor_item.setVisible(False)
            self.cursor_item2.setVisible(False)
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = False
            self._apply_tool_cursor()
            return
        super().keyReleaseEvent(e)

    def _apply_tool_cursor(self):
        self.viewport().setCursor(Qt.OpenHandCursor if self.tool == "pan" else Qt.CrossCursor)

    def set_tool(self, tool: str):
        self.tool = tool
        self._apply_tool_cursor()

    def mousePressEvent(self, e):
        sp = self.mapToScene(e.position().toPoint())
        if e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and (self._space or self.tool == "pan")):
            self._pan = e.position()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            return
        if not self._has_image:
            return
        if e.button() == Qt.RightButton:
            self.contextRequested.emit(sp.x(), sp.y(), e.globalPosition().toPoint())
            return
        if e.button() == Qt.LeftButton:
            if self.tool == "spot":
                self.spotRequested.emit([(sp.x(), sp.y())], self.radius)
            elif self.tool == "brush":
                self._stroke = [(sp.x(), sp.y())]
                self._update_stroke()

    def mouseMoveEvent(self, e):
        sp = self.mapToScene(e.position().toPoint())
        self.cursorMoved.emit(sp.x(), sp.y())
        if self._pan is not None:
            d = e.position() - self._pan
            self._pan = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        self._update_cursor(sp)
        if self._stroke is not None:
            lx, ly = self._stroke[-1]
            step = max(1.0, self.radius * 0.35)
            if (sp.x() - lx) ** 2 + (sp.y() - ly) ** 2 >= step * step:
                self._stroke.append((sp.x(), sp.y()))
                self._update_stroke()

    def mouseReleaseEvent(self, e):
        if self._pan is not None and e.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._pan = None
            self.viewport().setCursor(Qt.OpenHandCursor if (self._space or self.tool == "pan") else Qt.CrossCursor)
            return
        if self._stroke is not None and e.button() == Qt.LeftButton:
            pts = self._stroke
            self._stroke = None
            self.stroke_item.setPath(QPainterPath())
            self.spotRequested.emit(pts, self.radius)

    def leaveEvent(self, e):
        self.cursor_item.setVisible(False)
        self.cursor_item2.setVisible(False)
        super().leaveEvent(e)

    def _update_stroke(self):
        path = QPainterPath()
        for x, y in self._stroke:
            path.addEllipse(QPointF(x, y), self.radius, self.radius)
        path = path.simplified()
        self.stroke_item.setPath(path)
        self.stroke_item.setBrush(QBrush(QColor(255, 80, 80, 70)))
        pen = QPen(QColor(255, 80, 80), 1)
        pen.setCosmetic(True)
        self.stroke_item.setPen(pen)

    def resizeEvent(self, e):
        super().resizeEvent(e)
