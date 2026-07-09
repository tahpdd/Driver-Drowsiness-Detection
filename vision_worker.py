from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Signal

BASE_DIR = Path(__file__).resolve().parent

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError as exc:
    raise SystemExit("[ERR] pip install mediapipe") from exc

try:
    import camera as camera_mod
    from ear import LEFT_EYE, RIGHT_EYE, EyeTracker
    from mar import LABELS, LIP, MarPipeline
except ImportError as exc:
    raise SystemExit("[ERR] ddd_ui logic modules were not found.") from exc


MODEL_DIR = BASE_DIR / "models"

LANDMARK_GROUPS = {
    "left_eye": tuple(dict.fromkeys(LEFT_EYE.values())),
    "right_eye": tuple(dict.fromkeys(RIGHT_EYE.values())),
    "mouth": tuple(dict.fromkeys(LIP.values())),
}


def find_model(filename: str) -> Path:
    path = MODEL_DIR / filename
    if path.exists():
        return path
    raise FileNotFoundError(f"Missing model {filename}. Expected: {path}")


def collect_tracked_landmarks(lms) -> dict[str, list[tuple[float, float]]]:
    if lms is None:
        return {name: [] for name in LANDMARK_GROUPS}
    return {
        name: [(float(lms[idx].x), float(lms[idx].y)) for idx in indexes]
        for name, indexes in LANDMARK_GROUPS.items()
    }


class VisionWorker(QThread):
    initialized = Signal(object)
    frame_ready = Signal(object)
    error = Signal(str)
    stopped = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop_requested = False
        self.stream = None
        self.detector = None
        self.eye_tracker: EyeTracker | None = None
        self.mar_pipeline: MarPipeline | None = None

    def stop(self) -> None:
        self._stop_requested = True
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass

    def run(self) -> None:
        frame_idx = 0

        try:
            mp_model = find_model("face_landmarker.task")
            onnx_model = find_model("tcn_model.onnx")

            opts = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(mp_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
            )
            self.detector = vision.FaceLandmarker.create_from_options(opts)
            self.mar_pipeline = MarPipeline(str(onnx_model))
            self.eye_tracker = EyeTracker()
            self.stream = camera_mod.CameraStream(
                camera_mod.CAM_W,
                camera_mod.CAM_H,
                camera_mod.CAM_FPS,
            )

            self.initialized.emit(
                {
                    "camera_source": self.stream.source_label,
                    "mp_model": str(mp_model),
                    "onnx_model": str(onnx_model),
                    "mar_input": self.mar_pipeline.input_name,
                    "mar_input_shape": self.mar_pipeline.input_shape,
                    "labels": list(LABELS),
                }
            )

            fps_dt: deque[float] = deque(maxlen=30)
            prev_frame_t: float | None = None
            prev_mp_ts_ms: int | None = None

            while not self._stop_requested:
                ret, frame_rgb = self.stream.read()
                if self._stop_requested:
                    break
                if not ret:
                    raise RuntimeError("Camera read failed.")

                frame_idx += 1
                now = time.monotonic()

                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                ts_ms = int(now * 1000.0)
                if prev_mp_ts_ms is not None and ts_ms <= prev_mp_ts_ms:
                    ts_ms = prev_mp_ts_ms + 1
                prev_mp_ts_ms = ts_ms

                t0 = time.perf_counter()
                result = self.detector.detect_for_video(mp_img, ts_ms)
                t1 = time.perf_counter()
                mp_ms = (t1 - t0) * 1000.0

                if prev_frame_t is not None:
                    fps_dt.append(now - prev_frame_t)
                prev_frame_t = now
                fps = (1.0 / (sum(fps_dt) / len(fps_dt))) if fps_dt else 0.0

                face_ok = bool(result.face_landmarks)
                lms = result.face_landmarks[0] if face_ok else None
                landmarks = collect_tracked_landmarks(lms)
                visibility = {
                    "left_eye": face_ok,
                    "right_eye": face_ok,
                    "mouth": face_ok,
                }

                eye_state = self.eye_tracker.update(face_ok, lms, now)
                mar_state = self.mar_pipeline.update(
                    face_ok,
                    lms,
                    now,
                    clock=time.perf_counter,
                )

                self.frame_ready.emit(
                    {
                        "frame_idx": frame_idx,
                        "frame_rgb": frame_rgb,
                        "landmarks": landmarks,
                        "face_ok": face_ok,
                        "fps": fps,
                        "mp_ms": mp_ms,
                        "visibility": visibility,
                        "mar": mar_state,
                        "eye": eye_state,
                    }
                )

        except Exception as exc:
            self.error.emit(str(exc))

        finally:
            now = time.monotonic()
            if self.eye_tracker is not None:
                try:
                    self.eye_tracker.finalize(now)
                except Exception:
                    pass

            if self.detector is not None:
                try:
                    self.detector.close()
                except Exception:
                    pass

            if self.stream is not None:
                try:
                    self.stream.stop()
                except Exception:
                    pass

            summary: dict[str, Any] = {
                "frames": frame_idx,
            }
            if self.mar_pipeline is not None:
                summary.update(
                    {
                        "mouth_events": self.mar_pipeline.event_count,
                        "yawns": self.mar_pipeline.yawn_cnt,
                        "talks": self.mar_pipeline.talk_cnt,
                    }
                )
            if self.eye_tracker is not None:
                any_alert_2s = any(evt.alert_2s for evt in self.eye_tracker.events)
                summary.update(
                    {
                        "eye_events": len(self.eye_tracker.events),
                        "max_eye_run": self.eye_tracker.max_bilateral_closed_run_s,
                        "any_eye_alert": any_alert_2s,
                    }
                )
            self.stopped.emit(summary)
