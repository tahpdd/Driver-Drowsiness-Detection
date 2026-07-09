from __future__ import annotations

import os
import platform
import queue
import re
import subprocess
import threading

import numpy as np

CAM_W = 1280
CAM_H = 720
CAM_FPS = 30

LINUX_CAM_DEV = "/dev/video2"

WIN_CAM_NAME = os.environ.get("CAMERA_NAME", "").strip()
WIN_CAM_INDEX = os.environ.get("CAMERA_INDEX", "").strip()


def _decode_process_text(data: bytes) -> str:
    for enc in ("utf-8", "mbcs"):
        try:
            return data.decode(enc, errors="replace")
        except LookupError:
            continue
    return data.decode(errors="replace")


def list_windows_cameras() -> list[str]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-list_devices",
        "true",
        "-f",
        "dshow",
        "-i",
        "dummy",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg was not found. Install ffmpeg or add it to PATH.") from exc
    text = _decode_process_text(proc.stdout) + _decode_process_text(proc.stderr)

    devices: list[str] = []
    video_re = re.compile(r'"([^"]+)"\s+\(video\)')

    for line in text.splitlines():
        video_match = video_re.search(line)
        if video_match:
            devices.append(video_match.group(1))

    return devices


def _choose_windows_camera() -> str:
    if WIN_CAM_NAME:
        return WIN_CAM_NAME

    devices = list_windows_cameras()
    if not devices:
        raise RuntimeError(
            "No Windows camera found by ffmpeg dshow. "
            "Check ffmpeg and camera permission, or set CAMERA_NAME."
        )

    if WIN_CAM_INDEX:
        if not WIN_CAM_INDEX.isdigit():
            raise RuntimeError("CAMERA_INDEX must be a number.")
        idx = int(WIN_CAM_INDEX)
        if 0 <= idx < len(devices):
            return devices[idx]
        raise RuntimeError(f"CAMERA_INDEX={idx} is out of range 0..{len(devices) - 1}.")

    return devices[0]


class _RawFFmpegStream:
    def __init__(self, cmd: list[str], w: int, h: int, source_label: str):
        self.w = w
        self.h = h
        self.source_label = source_label
        self.frame_size = w * h * 3
        self.stopped = False
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.frame_size * 2,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg was not found. Install ffmpeg or add it to PATH.") from exc
        self.q: queue.Queue[bytes] = queue.Queue(maxsize=1)
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        while not self.stopped:
            try:
                if self.proc.stdout is None:
                    break
                raw = self.proc.stdout.read(self.frame_size)
            except Exception:
                break
            if not raw or len(raw) != self.frame_size:
                self.stopped = True
                break
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(raw)

    def read(self):
        if self.stopped:
            return False, None
        try:
            raw = self.q.get(timeout=5)
        except queue.Empty:
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.h, self.w, 3))
        return True, frame

    def stop(self) -> None:
        self.stopped = True
        try:
            if self.proc.stdout is not None:
                self.proc.stdout.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()
            self.proc.wait()


def _linux_cmd(dev: str, w: int, h: int, fps: int) -> list[str]:
    return [
        "ffmpeg",
        "-loglevel",
        "quiet",
        "-f",
        "v4l2",
        "-input_format",
        "mjpeg",
        "-video_size",
        f"{w}x{h}",
        "-framerate",
        str(fps),
        "-i",
        dev,
        "-f",
        "image2pipe",
        "-pix_fmt",
        "rgb24",
        "-vcodec",
        "rawvideo",
        "-",
    ]


def _windows_cmd(camera_name: str, w: int, h: int, fps: int) -> list[str]:
    return [
        "ffmpeg",
        "-loglevel",
        "quiet",
        "-f",
        "dshow",
        "-i",
        f"video={camera_name}",
        "-vf",
        f"scale={w}:{h}",
        "-r",
        str(fps),
        "-f",
        "image2pipe",
        "-pix_fmt",
        "rgb24",
        "-vcodec",
        "rawvideo",
        "-",
    ]


class CameraStream:
    def __init__(self, w: int = CAM_W, h: int = CAM_H, fps: int = CAM_FPS):
        system = platform.system().lower()
        if system == "windows":
            camera_name = _choose_windows_camera()
            cmd = _windows_cmd(camera_name, w, h, fps)
            self._stream = _RawFFmpegStream(
                cmd,
                w,
                h,
                source_label=f"Windows dshow: {camera_name}",
            )
        else:
            cmd = _linux_cmd(LINUX_CAM_DEV, w, h, fps)
            self._stream = _RawFFmpegStream(
                cmd,
                w,
                h,
                source_label=f"Linux v4l2: {LINUX_CAM_DEV}",
            )

    @property
    def source_label(self) -> str:
        return self._stream.source_label

    def read(self):
        return self._stream.read()

    def stop(self) -> None:
        self._stream.stop()
