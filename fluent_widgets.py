from __future__ import annotations

import math
import time
from collections import deque
from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


LABELS = ["Normal", "Talking", "Yawning"]
LANDMARK_COLORS = {
    "left_eye": "#3b82f6",
    "right_eye": "#10b981",
    "mouth": "#ef4444",
}
TONE_COLORS = {
    "normal": "#0f172a",
    "muted": "#64748b",
    "good": "#10b981",
    "warn": "#f59e0b",
    "bad": "#ef4444",
    "accent": "#3b82f6",
}
TONE_BACKGROUNDS = {
    "normal": "#475569",
    "muted": "#64748b",
    "good": "#10b981",
    "warn": "#f59e0b",
    "bad": "#ef4444",
    "accent": "#3b82f6",
}


APP_STYLESHEET = """
QWidget {
    color: #0f172a;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 12px;
}
QWidget#Shell {
    background: #e2e8f0;
}
QWidget#MainView,
QWidget#Sidebar {
    background: transparent;
}
QFrame#Panel,
QFrame#MetricCard {
    background: #ffffff;
    border: 1px solid #94a3b8;
    border-radius: 4px;
}
QFrame#StatusStrip {
    background: #000000;
    border: 1px solid #000000;
    border-radius: 0;
}
QFrame#PanelHeader,
QFrame#CardHeader {
    background: transparent;
    border: 0;
    border-bottom: 1px solid #94a3b8;
}
QLabel#PanelTitle,
QLabel#CardTitle {
    color: #0f172a;
    font-size: 11px;
    font-weight: 700;
}
QLabel#StatusBadge {
    color: #ffffff;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 2px;
}
QWidget#CameraSurface {
    background: #000000;
    border: 1px solid #000000;
    border-radius: 0;
}
QFrame#DataRow {
    background: transparent;
    border: 0;
    border-bottom: 1px dotted #cbd5e1;
}
QLabel#RowKey {
    color: #64748b;
    font-size: 11px;
}
QLabel#RowValue {
    color: #0f172a;
    font-family: "Consolas", "Cascadia Mono", monospace;
    font-size: 11px;
    font-weight: 700;
}
QLabel#SectionHint {
    color: #0f172a;
    font-size: 11px;
    font-weight: 700;
}
QProgressBar {
    min-height: 5px;
    max-height: 5px;
    border: 0;
    border-radius: 0;
    background: #e2e8f0;
}
QProgressBar::chunk {
    border-radius: 0;
}
QTabWidget#InfoTabs::pane {
    border: 0;
    background: transparent;
}
QTabBar::tab {
    background: #cbd5e1;
    color: #334155;
    font-size: 11px;
    font-weight: 700;
    min-width: 82px;
    padding: 5px 10px;
    border: 1px solid #94a3b8;
    border-bottom: 0;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #0f172a;
}
QFrame#AlertZone {
    background: #e2e8f0;
    border: 1px solid #94a3b8;
    border-radius: 4px;
}
QLabel#AlertTitle {
    color: #334155;
    font-size: 11px;
    font-weight: 700;
}
QLabel#AlertState {
    color: #0f172a;
    font-size: 21px;
    font-weight: 800;
}
QLabel#AlertDetail,
QLabel#AlertWarning,
QLabel#AlertFooter {
    color: #0f172a;
    font-size: 12px;
    font-weight: 700;
}
"""


def _tone_color(tone: str) -> str:
    return TONE_COLORS.get(tone, TONE_COLORS["normal"])


def _tone_background(tone: str) -> str:
    return TONE_BACKGROUNDS.get(tone, TONE_BACKGROUNDS["normal"])


def _tone_surface(tone: str) -> str:
    return {
        "muted": "#cbd5e1",
        "good": "#4ade80",
        "warn": "#fbbf24",
        "bad": "#f87171",
        "accent": "#60a5fa",
    }.get(tone, "#e2e8f0")


def _tone_border(tone: str) -> str:
    return {
        "muted": "#64748b",
        "good": "#15803d",
        "warn": "#b45309",
        "bad": "#b91c1c",
        "accent": "#1d4ed8",
    }.get(tone, "#94a3b8")


def _fmt(value: float | None, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "---"
    return f"{value:.{digits}f}{suffix}"


def _compact_text(value: Any, max_chars: int = 30) -> str:
    text = str(value) if value not in (None, "") else "---"
    text = text.replace("\\", "/")
    if len(text) <= max_chars:
        return text
    return "..." + text[-(max_chars - 3) :]


def eye_tone(state: str) -> str:
    if state == "OPEN":
        return "good"
    if state in {"CLOSED", "BOTH_CLOSED"}:
        return "bad"
    if state == "NO_FACE":
        return "muted"
    return "normal"


def pipe_tone(pipe: str) -> str:
    return {
        "IDLE": "muted",
        "MONITORING": "warn",
        "RECORDING": "warn",
        "INFER": "warn",
    }.get(pipe, "normal")


class StatusBadge(QLabel):
    def __init__(self, text: str = "STARTING", tone: str = "warn", parent=None):
        super().__init__(text, parent)
        self.setObjectName("StatusBadge")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(18)
        self.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        self.set_value(text, tone)

    def set_value(self, text: str, tone: str = "normal") -> None:
        self.setText(text.upper())
        self.setStyleSheet(
            "QLabel#StatusBadge {"
            f"background: {_tone_background(tone)};"
            "color: #ffffff;"
            "font-size: 10px;"
            "font-weight: 700;"
            "padding: 2px 6px;"
            "border-radius: 2px;"
            "}"
        )


class PanelFrame(QFrame):
    def __init__(
        self,
        title: str,
        badge_text: str | None = None,
        badge_tone: str = "normal",
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("Panel")

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(4)

        header = QFrame()
        header.setObjectName("PanelHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 3)
        header_layout.setSpacing(8)

        title_label = QLabel(title.upper())
        title_label.setObjectName("PanelTitle")
        header_layout.addWidget(title_label, 1)

        self.badge: StatusBadge | None = None
        if badge_text is not None:
            self.badge = StatusBadge(badge_text, badge_tone)
            header_layout.addWidget(self.badge, 0)

        self._layout.addWidget(header, 0)

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)
        self._layout.addWidget(self.content, 1)

    def set_badge(self, text: str, tone: str = "normal") -> None:
        if self.badge is not None:
            self.badge.set_value(text, tone)


class CameraView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: QImage | None = None
        self._landmarks: dict[str, list[tuple[float, float]]] = {}
        self.setObjectName("CameraSurface")
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def set_frame(
        self,
        frame_rgb: Any,
        landmarks: dict[str, list[tuple[float, float]]],
    ) -> None:
        height, width, channels = frame_rgb.shape
        if channels != 3:
            return
        bytes_per_line = width * channels
        self._image = QImage(
            frame_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        self._landmarks = landmarks or {}
        self.update()

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#000000"))

        if self._image is None:
            painter.end()
            return

        iw = self._image.width()
        ih = self._image.height()
        if iw <= 0 or ih <= 0:
            painter.end()
            return

        rw = self.width()
        rh = self.height()
        scale = min(rw / iw, rh / ih)
        tw = iw * scale
        th = ih * scale
        left = (rw - tw) / 2.0
        top = (rh - th) / 2.0
        target = QRectF(left, top, tw, th)

        painter.drawImage(target, self._image, QRectF(0, 0, iw, ih))

        radius = max(1.15, min(2.0, tw / 740.0))
        for group_name, points in self._landmarks.items():
            color = QColor(LANDMARK_COLORS.get(group_name, "#38bdf8"))
            color.setAlpha(215)
            painter.setPen(QPen(QColor(255, 255, 255, 165), 0.6))
            painter.setBrush(color)
            for x_norm, y_norm in points:
                if 0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0:
                    painter.drawEllipse(
                        QPointF(left + x_norm * tw, top + y_norm * th),
                        radius,
                        radius,
                    )

        painter.end()


class SignalChart(QWidget):
    def __init__(
        self,
        label: str,
        color: str,
        y_min: float,
        y_max: float,
        max_samples: int = 180,
        parent=None,
    ):
        super().__init__(parent)
        self.label = label
        self.color = QColor(color)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.values: deque[float] = deque(maxlen=max_samples)
        self.setMinimumHeight(40)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def add_sample(self, value: float | None) -> None:
        if value is None or not math.isfinite(float(value)):
            return
        self.values.append(float(value))
        self.update()

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect()
        painter.fillRect(rect, QColor("#1e1e1e"))

        grid_pen = QPen(QColor(255, 255, 255, 28), 1)
        painter.setPen(grid_pen)
        for x in range(0, max(1, rect.width()), 20):
            painter.drawLine(x, 0, x, rect.height())
        for y in range(0, max(1, rect.height()), 20):
            painter.drawLine(0, y, rect.width(), y)

        painter.fillRect(5, 4, max(54, len(self.label) * 6), 15, QColor(0, 0, 0, 165))
        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("Consolas", 7))
        painter.drawText(8, 15, self.label)

        if len(self.values) < 2:
            painter.end()
            return

        plot = QRectF(6, 20, rect.width() - 12, rect.height() - 26)
        if plot.width() <= 1 or plot.height() <= 1:
            painter.end()
            return

        span = max(self.y_max - self.y_min, 1e-6)
        points = []
        values = list(self.values)
        for idx, value in enumerate(values):
            x = plot.left() + (idx / (len(values) - 1)) * plot.width()
            normalized = (max(self.y_min, min(self.y_max, value)) - self.y_min) / span
            y = plot.bottom() - normalized * plot.height()
            points.append(QPointF(x, y))

        painter.setPen(QPen(self.color, 1.5))
        painter.drawPolyline(QPolygonF(points))
        painter.end()


class DualSignalChart(QWidget):
    def __init__(
        self,
        label: str,
        first_label: str,
        first_color: str,
        second_label: str,
        second_color: str,
        y_min: float,
        y_max: float,
        max_samples: int = 180,
        parent=None,
    ):
        super().__init__(parent)
        self.label = label
        self.first_label = first_label
        self.second_label = second_label
        self.first_color = QColor(first_color)
        self.second_color = QColor(second_color)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.first_values: deque[float] = deque(maxlen=max_samples)
        self.second_values: deque[float] = deque(maxlen=max_samples)
        self.setMinimumHeight(40)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def add_samples(self, first: float | None, second: float | None) -> None:
        if first is not None and math.isfinite(float(first)):
            self.first_values.append(float(first))
        if second is not None and math.isfinite(float(second)):
            self.second_values.append(float(second))
        self.update()

    def _draw_series(
        self,
        painter: QPainter,
        plot: QRectF,
        values: deque[float],
        color: QColor,
    ) -> None:
        if len(values) < 2:
            return
        span = max(self.y_max - self.y_min, 1e-6)
        points = []
        series = list(values)
        for idx, value in enumerate(series):
            x = plot.left() + (idx / (len(series) - 1)) * plot.width()
            normalized = (max(self.y_min, min(self.y_max, value)) - self.y_min) / span
            y = plot.bottom() - normalized * plot.height()
            points.append(QPointF(x, y))
        painter.setPen(QPen(color, 1.5))
        painter.drawPolyline(QPolygonF(points))

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect()
        painter.fillRect(rect, QColor("#1e1e1e"))

        grid_pen = QPen(QColor(255, 255, 255, 28), 1)
        painter.setPen(grid_pen)
        for x in range(0, max(1, rect.width()), 20):
            painter.drawLine(x, 0, x, rect.height())
        for y in range(0, max(1, rect.height()), 20):
            painter.drawLine(0, y, rect.width(), y)

        painter.fillRect(5, 4, 118, 15, QColor(0, 0, 0, 165))
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(8, 15, self.label)
        painter.setPen(self.first_color)
        painter.drawText(64, 15, self.first_label)
        painter.setPen(self.second_color)
        painter.drawText(91, 15, self.second_label)

        plot = QRectF(6, 20, rect.width() - 12, rect.height() - 26)
        if plot.width() > 1 and plot.height() > 1:
            self._draw_series(painter, plot, self.first_values, self.first_color)
            self._draw_series(painter, plot, self.second_values, self.second_color)

        painter.end()


class MetricCard(QFrame):
    def __init__(
        self,
        title: str,
        badge_text: str | None = None,
        badge_tone: str = "normal",
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("MetricCard")
        self._rows: dict[str, QLabel] = {}
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(7, 5, 7, 5)
        self._layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("CardHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 3)
        header_layout.setSpacing(8)

        title_label = QLabel(title.upper())
        title_label.setObjectName("CardTitle")
        header_layout.addWidget(title_label, 1)

        self.badge: StatusBadge | None = None
        if badge_text is not None:
            self.badge = StatusBadge(badge_text, badge_tone)
            header_layout.addWidget(self.badge, 0)

        self._layout.addWidget(header)

    def add_row(self, key: str, label: str) -> None:
        row = QFrame()
        row.setObjectName("DataRow")
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 1, 0, 1)
        row_layout.setSpacing(6)

        key_label = QLabel(label)
        key_label.setObjectName("RowKey")
        key_label.setFixedWidth(78)
        key_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        value_label = QLabel("---")
        value_label.setObjectName("RowValue")
        value_label.setWordWrap(False)
        value_label.setMinimumWidth(0)
        value_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        value_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        row_layout.addWidget(key_label)
        row_layout.addWidget(value_label, 1)
        self._layout.addWidget(row)
        self._rows[key] = value_label

    def set_value(self, key: str, value: str, tone: str = "normal") -> None:
        label = self._rows.get(key)
        if label is None:
            return
        fm = QFontMetrics(label.font())
        target_width = label.width() if label.width() > 32 else 210
        label.setToolTip(str(value))
        label.setText(fm.elidedText(str(value), Qt.TextElideMode.ElideLeft, target_width))
        label.setStyleSheet(f"color: {_tone_color(tone)}; font-weight: 700;")


class AlertZone(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("AlertZone")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        self.title = QLabel(title.upper())
        self.title.setObjectName("AlertTitle")
        self.state = QLabel("WAITING")
        self.state.setObjectName("AlertState")
        self.detail = QLabel("---")
        self.detail.setObjectName("AlertDetail")
        self.warning = QLabel("---")
        self.warning.setObjectName("AlertWarning")
        self.footer = QLabel("---")
        self.footer.setObjectName("AlertFooter")

        for label in (self.title, self.state, self.detail, self.warning, self.footer):
            label.setMinimumWidth(0)
            label.setWordWrap(True)
            layout.addWidget(label)

        layout.addStretch(1)
        self.set_state("WAITING", "---", "---", "---", "muted")

    def set_state(
        self,
        state: str,
        detail: str,
        warning: str,
        footer: str,
        tone: str,
        warning_tone: str = "muted",
    ) -> None:
        self.setStyleSheet(
            "QFrame#AlertZone {"
            f"background: {_tone_surface(tone)};"
            f"border: 1px solid {_tone_border(tone)};"
            "border-radius: 4px;"
            "}"
        )
        self.state.setText(state.upper())
        self.detail.setText(detail)
        self.warning.setText(warning)
        self.footer.setText(footer)
        self.warning.setStyleSheet(f"color: {_tone_color(warning_tone)}; font-weight: 800;")


class AlertPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(352)
        self._mouth_result_seq = 0
        self._mouth_result_until_s = 0.0
        self._mouth_result_state = "---"
        self._mouth_result_tone = "muted"
        self._mouth_result_detail = "---"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.eye_zone = AlertZone("Eye Alert")
        self.mouth_zone = AlertZone("Mouth Alert")
        layout.addWidget(self.eye_zone, 1)
        layout.addWidget(self.mouth_zone, 1)

    def update_metrics(self, payload: dict[str, Any]) -> None:
        self._update_eye(payload)
        self._update_mouth(payload)

    def _update_eye(self, payload: dict[str, Any]) -> None:
        eye = payload["eye"]
        if eye.alert_on:
            state = eye.overall_state
            tone = "bad"
            detail = f"Closed {eye.bilateral_closed_run_s:.2f}s"
            warning = "Alert >= 2.0s: YES"
            warning_tone = "bad"
        elif eye.overall_state == "NO_FACE" or not payload.get("face_ok", False):
            state = "No face"
            tone = "muted"
            detail = f"Lost {eye.face_lost_s:.1f}s"
            warning = "Alert >= 2.0s: NO"
            warning_tone = "muted"
        else:
            state = eye.overall_state
            tone = eye_tone(eye.overall_state)
            detail = f"Closed {eye.bilateral_closed_run_s:.2f}s"
            warning = "Alert >= 2.0s: NO"
            warning_tone = "muted"

        self.eye_zone.set_state(state, detail, warning, "", tone, warning_tone)

    def _update_mouth(self, payload: dict[str, Any]) -> None:
        mar = payload["mar"]
        display_pred = int(getattr(mar, "display_pred", -1))
        result_seq = int(getattr(mar, "result_seq", 0))
        face_ok = bool(payload.get("face_ok", False))
        now_s = time.monotonic()

        if result_seq != self._mouth_result_seq and 0 <= display_pred < len(LABELS):
            self._mouth_result_seq = result_seq
            self._mouth_result_until_s = now_s + 2.0
            self._mouth_result_state = LABELS[display_pred]
            self._mouth_result_tone = ["good", "warn", "bad"][display_pred]
            self._mouth_result_detail = f"MAR {mar.mar:.3f}"

        if not face_ok or mar.pipe == "IDLE":
            state = "---"
            tone = "muted"
            detail = "No face"
        elif mar.pipe in {"RECORDING", "INFER"}:
            state = "---"
            tone = "muted"
            detail = f"Recording {mar.rec_dur:.1f}s | {mar.rec_frames} frames"
        elif now_s <= self._mouth_result_until_s:
            state = self._mouth_result_state
            tone = self._mouth_result_tone
            detail = self._mouth_result_detail
        else:
            state = "---"
            tone = "muted"
            detail = f"MAR {mar.mar:.3f}"

        if mar.yawn_alert_on:
            warning = f"15m warning: YES ({mar.yawn_15m_cnt}/3)"
            warning_tone = "bad"
        else:
            warning = f"15m warning: NO ({mar.yawn_15m_cnt}/3)"
            warning_tone = "muted"

        footer = f"Yawning count: {mar.yawn_cnt}"
        self.mouth_zone.set_state(state, detail, warning, footer, tone, warning_tone)


class SystemStatusBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("StatusStrip")
        self._source_label = "---"
        self.setFixedHeight(32)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(0)

        self.line = QLabel(
            "FPS: -- | Face: -- | Ready: -- | Eye L: -- R: -- | Mouth: -- | "
            "MP: -- ms | TCN: -- ms | Frame: -- | Msg: Starting"
        )
        self.line.setObjectName("RowValue")
        self.line.setTextFormat(Qt.TextFormat.RichText)
        self.line.setMinimumWidth(0)
        self.line.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.line.setStyleSheet("color: #ffffff; font-weight: 700;")
        layout.addWidget(self.line, 1)

    def _yn(self, value: bool) -> str:
        text = "YES" if value else "NO"
        color = "#22c55e" if value else "#ef4444"
        return f"<span style='color:{color};'>{text}</span>"

    def _set_line(self, html: str, tooltip: str | None = None) -> None:
        self.line.setToolTip(tooltip or html.replace("<span style='color:#22c55e;'>", "").replace("<span style='color:#ef4444;'>", "").replace("</span>", ""))
        self.line.setText(f"<span style='color:#ffffff;'>{html}</span>")

    def set_runtime_status(self, text: str, tone: str) -> None:
        color = _tone_color(tone)
        self._set_line(f"Msg: <span style='color:{color};'>{text}</span>", text)

    def set_initialized(self, payload: dict[str, Any]) -> None:
        self._source_label = _compact_text(payload.get("camera_source", "---"), 26)
        self._set_line(f"Source: {self._source_label} | Msg: <span style='color:#22c55e;'>Running</span>")

    def set_error(self, message: str) -> None:
        self._set_line(f"Msg: <span style='color:#ef4444;'>{_compact_text(message, 120)}</span>", message)

    def set_stopped(self, summary: dict[str, Any]) -> None:
        frames = summary.get("frames", 0)
        mouth_events = summary.get("mouth_events", 0)
        yawns = summary.get("yawns", 0)
        talks = summary.get("talks", 0)
        self._set_line(
            f"Frames: {frames} | Mouth events: {mouth_events} | Yawn: {yawns} | Talk: {talks} | "
            "<span style='color:#94a3b8;'>Msg: Stopped</span>"
        )

    def update_metrics(self, payload: dict[str, Any]) -> None:
        mar = payload["mar"]
        eye = payload["eye"]
        visibility = payload.get("visibility", {})
        eye_l = bool(visibility.get("left_eye", False))
        eye_r = bool(visibility.get("right_eye", False))
        mouth = bool(visibility.get("mouth", False))
        html = (
            f"FPS: {payload['fps']:4.1f} | "
            f"Face: {'Y' if payload['face_ok'] else 'N'} | "
            f"Ready: {'Y' if eye.ready else 'N'} | "
            f"Eye L: {self._yn(eye_l)} R: {self._yn(eye_r)} | "
            f"Mouth: {self._yn(mouth)} | "
            f"MP: {payload['mp_ms']:4.1f} ms | "
            f"TCN: {mar.tcn_ms:4.1f} ms | "
            f"Frame: {payload['frame_idx']:06d} | "
            "Msg: Running"
        )
        tooltip = (
            f"FPS: {payload['fps']:4.1f} | Face: {payload['face_ok']} | Ready: {eye.ready} | "
            f"Eye L: {eye_l} R: {eye_r} | Mouth: {mouth} | "
            f"MP: {payload['mp_ms']:4.1f} ms | TCN: {mar.tcn_ms:4.1f} ms | "
            f"Frame: {payload['frame_idx']:06d} | Source: {self._source_label}"
        )
        self._set_line(html, tooltip)


class InfoPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(352)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        self.cards = outer

        self.mouth = MetricCard("Mouth Parameters")
        for key, label in (
            ("mar_pipe", "MAR / Pipe"),
            ("rec_pred", "Rec / Pred"),
            ("prob", "TCN % N/T/Y"),
            ("counts", "Y/T/15m"),
            ("alert", "Alert"),
        ):
            self.mouth.add_row(key, label)
        self.cards.addWidget(self.mouth)

        self.eye = MetricCard("Eye Parameters")
        for key, label in (
            ("state_alert", "State / Alert"),
            ("closed_run", "Closed / Max"),
            ("side_state", "Eye State"),
            ("ear", "EAR L/R"),
            ("open_pct", "Open % L/R"),
            ("ref", "Ref L/R"),
            ("misc", "Samp/Lost/Asym"),
        ):
            self.eye.add_row(key, label)
        self.cards.addWidget(self.eye)

        self.cards.addStretch(1)

    def update_metrics(self, payload: dict[str, Any]) -> None:
        mar = payload["mar"]
        eye = payload["eye"]

        self.mouth.set_value("mar_pipe", f"{mar.mar:.3f} / {mar.pipe}", pipe_tone(mar.pipe))

        if mar.pipe == "RECORDING":
            self.mouth.set_value("rec_pred", f"{mar.rec_dur:.1f}s/{mar.rec_frames} | WAIT", "warn")
            self.mouth.set_value("prob", "0/0/0", "muted")
        elif mar.pred >= 0 and mar.probs:
            self.mouth.set_value(
                "rec_pred",
                f"{mar.rec_dur:.1f}s/{mar.rec_frames} | {LABELS[mar.pred].upper()}",
                ["good", "warn", "bad"][mar.pred],
            )
            probs = [mar.probs[idx] * 100.0 if idx < len(mar.probs) else 0.0 for idx in range(3)]
            self.mouth.set_value("prob", f"{probs[0]:.0f}/{probs[1]:.0f}/{probs[2]:.0f}", "normal")
        else:
            self.mouth.set_value("rec_pred", f"{mar.rec_dur:.1f}s/{mar.rec_frames} | N/A", "muted")
            self.mouth.set_value("prob", "--/--/--", "muted")
        self.mouth.set_value("counts", f"{mar.yawn_cnt}/{mar.talk_cnt}/{mar.yawn_15m_cnt}", "normal")
        self.mouth.set_value("alert", "YES" if mar.yawn_alert_on else "NO", "bad" if mar.yawn_alert_on else "muted")

        left = eye.left
        right = eye.right
        self.eye.set_value(
            "state_alert",
            f"{eye.overall_state} / {'YES' if eye.alert_on else 'NO'}",
            "bad" if eye.alert_on else eye_tone(eye.overall_state),
        )
        self.eye.set_value(
            "closed_run",
            f"{eye.bilateral_closed_run_s:.2f}s / {eye.max_bilateral_closed_run_s:.2f}s",
            "bad" if eye.alert_on else "normal",
        )
        self.eye.set_value("side_state", eye.overall_state, eye_tone(eye.overall_state))
        self.eye.set_value("ear", f"{left.smooth:.4f} / {right.smooth:.4f}", "normal")
        self.eye.set_value("open_pct", f"{left.open_pct:.0f} / {right.open_pct:.0f}", "normal")
        self.eye.set_value(
            "ref",
            f"{_fmt(left.open_ref, 3)}/{_fmt(left.close_ref, 3)} | {_fmt(right.open_ref, 3)}/{_fmt(right.close_ref, 3)}",
            "muted",
        )
        self.eye.set_value(
            "misc",
            f"{left.samples}/{right.samples} | {eye.face_lost_s:.1f}s | {eye.asym:.3f}",
            "warn" if eye.face_lost_s > 0 else "normal",
        )
