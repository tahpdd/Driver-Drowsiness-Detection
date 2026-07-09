from __future__ import annotations

import sys
from pathlib import Path

try:
    from PySide6.QtCore import QTimer, Qt, QUrl
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QTabWidget, QVBoxLayout, QWidget
except ImportError as exc:
    raise SystemExit(
        "[ERR] PySide6 is required. Run: pip install -r ddd_ui/requirements.txt"
    ) from exc

from fluent_widgets import (
    APP_STYLESHEET,
    AlertPanel,
    CameraView,
    DualSignalChart,
    InfoPanel,
    PanelFrame,
    SignalChart,
    SystemStatusBar,
    eye_tone,
    pipe_tone,
)
from vision_worker import VisionWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DDD Monitor")
        self.setFixedSize(1024, 600)
        self._closing = False
        self._had_error = False

        self._alarm_playing = False
        self._sound_path = Path(__file__).resolve().parent / "sound.mp4"
        self._sound_available = self._sound_path.exists()

        self._audio_output = QAudioOutput(self)
        self._audio_output.setVolume(1.0)

        self._alarm_player = QMediaPlayer(self)
        self._alarm_player.setAudioOutput(self._audio_output)

        if self._sound_available:
            self._alarm_player.setSource(QUrl.fromLocalFile(str(self._sound_path)))
            self._alarm_player.mediaStatusChanged.connect(self._on_alarm_media_status)
        else:
            print(f"[WARN] Sound file not found: {self._sound_path}")

        shell = QWidget()
        shell.setObjectName("Shell")
        layout = QVBoxLayout(shell)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)

        self.camera_panel = PanelFrame("Live Camera Feed", "STARTING", "warn")
        self.camera_view = CameraView()
        self.camera_panel.content_layout.addWidget(self.camera_view)

        right_view = QWidget()
        right_view.setObjectName("Sidebar")
        right_view.setFixedWidth(352)
        right_layout = QVBoxLayout(right_view)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        self.mar_panel = PanelFrame("Mouth Aspect Ratio", "MONITORING", "warn")
        self.mar_panel.setFixedHeight(102)
        self.mar_chart = SignalChart("MAR Signal", "#ef4444", 0.0, 2.5)
        self.mar_panel.content_layout.addWidget(self.mar_chart)

        self.ear_panel = PanelFrame("Eye Aspect Ratio", "TRACKING", "good")
        self.ear_panel.setFixedHeight(102)
        self.ear_chart = DualSignalChart(
            "EAR",
            "L",
            "#3b82f6",
            "R",
            "#10b981",
            0.0,
            0.5,
        )
        self.ear_panel.content_layout.addWidget(self.ear_chart)

        self.info_tabs = QTabWidget()
        self.info_tabs.setObjectName("InfoTabs")
        self.info_tabs.setFixedWidth(352)
        self.info_panel = InfoPanel()
        self.alert_panel = AlertPanel()
        self.info_tabs.addTab(self.info_panel, "DETAILS")
        self.info_tabs.addTab(self.alert_panel, "ALERTS")
        self.system_status = SystemStatusBar()

        right_layout.addWidget(self.mar_panel, 0)
        right_layout.addWidget(self.ear_panel, 0)
        right_layout.addWidget(self.info_tabs, 1)

        content_layout.addWidget(self.camera_panel, 1)
        content_layout.addWidget(right_view, 0)
        layout.addWidget(content, 1)
        layout.addWidget(self.system_status, 0)
        self.setCentralWidget(shell)

        self.worker = VisionWorker(self)
        self.worker.initialized.connect(self._on_initialized)
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.error.connect(self._on_error)
        self.worker.stopped.connect(self._on_stopped)

        QTimer.singleShot(150, self.worker.start)

    def _on_initialized(self, payload: dict) -> None:
        self.camera_panel.set_badge("LIVE", "good")
        self.system_status.set_initialized(payload)

    def _on_frame_ready(self, payload: dict) -> None:
        self.camera_view.set_frame(payload["frame_rgb"], payload["landmarks"])
        self.info_panel.update_metrics(payload)
        self.alert_panel.update_metrics(payload)
        self.system_status.update_metrics(payload)
        self.mar_chart.add_sample(payload["mar"].mar)
        self.ear_chart.add_samples(payload["eye"].left.smooth, payload["eye"].right.smooth)
        self.mar_panel.set_badge(payload["mar"].pipe, pipe_tone(payload["mar"].pipe))
        self.ear_panel.set_badge(payload["eye"].overall_state, eye_tone(payload["eye"].overall_state))

        self._update_eye_alarm(payload["eye"].alert_on)

    def _update_eye_alarm(self, should_play: bool) -> None:
        if not self._sound_available or not self._sound_path.exists():
            self._sound_available = False
            self._alarm_playing = False
            return

        if should_play and not self._alarm_playing:
            self._alarm_playing = True
            self._alarm_player.setPosition(0)
            self._alarm_player.play()

        elif not should_play and self._alarm_playing:
            self._alarm_playing = False
            self._alarm_player.stop()

    def _on_alarm_media_status(self, status) -> None:
        if (
            self._sound_available
            and self._alarm_playing
            and status == QMediaPlayer.MediaStatus.EndOfMedia
        ):
            self._alarm_player.setPosition(0)
            self._alarm_player.play()

    def _on_error(self, message: str) -> None:
        self._had_error = True
        self._update_eye_alarm(False)
        self.camera_panel.set_badge("ERROR", "bad")
        self.system_status.set_error(message)

    def _on_stopped(self, summary: dict) -> None:
        self._update_eye_alarm(False)
        self.camera_panel.set_badge("STOPPED", "muted")
        self.mar_panel.set_badge("STOPPED", "muted")
        self.ear_panel.set_badge("STOPPED", "muted")
        if not self._closing and not self._had_error:
            self.system_status.set_stopped(summary)

    def closeEvent(self, event):
        self._closing = True
        self._update_eye_alarm(False)

        if self.worker.isRunning():
            self.system_status.set_runtime_status("STOPPING", "warn")
            self.worker.stop()
            self.worker.wait(7000)
        event.accept()


def main() -> int:
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseStyleSheetPropagationInWidgetStyles, True)
    app = QApplication(sys.argv[:1])
    app.setStyleSheet(APP_STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
