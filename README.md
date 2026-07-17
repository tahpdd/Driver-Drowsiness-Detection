# DDD

## Overview

DDD is a real-time drowsiness detection application. MediaPipe tracks facial landmarks, EAR detects closed eyes, and MAR measures mouth movement. A Temporal Convolutional Network model uses the mouth data to detect yawning and talking. FFmpeg reads live video from the camera for the application.

## 1. Install Python libraries

```bash
pip install mediapipe numpy PySide6 onnxruntime
```

## 2. Install FFmpeg

Fedora Linux, Red Hat family

```bash
sudo dnf install ffmpeg-free
```

Ubuntu and Raspberry Pi OS, Debian family

```bash
sudo apt install ffmpeg
```

## 3. Set the camera on Linux

Open `camera.py` and change `/dev/video2` to your device camera.

## 4. Run

```bash
python main.py
```
