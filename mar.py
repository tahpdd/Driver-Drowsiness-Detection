from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

ONSET_THR = 0.25
ONSET_SUSTAIN_T = 0.10

OFFSET_THR = 0.15
OFFSET_SUSTAIN_T = 0.50
SLOPE_VAR_THR = 0.002
SLOPE_WIN_T = 0.50

MIN_RISE_T = 0.2

MIN_EVT_DUR = 2.5
MAX_EVT_DUR = 8.0
LOOKBACK_T = 0.50
STS_THR = 4.0

YAWN_ALERT_COUNT_15M = 3
YAWN_ALERT_WIN_S = 15.0 * 60.0

LABELS = ["Normal", "Talking", "Yawning"]

LIP = {
    "tc": 13,
    "tl": 82,
    "tr": 312,
    "cl": 78,
    "cr": 308,
    "bc": 14,
    "bl": 87,
    "br": 317,
}


def compute_mar(lms):
    def d(i, j):
        return math.hypot(lms[i].x - lms[j].x, lms[i].y - lms[j].y)

    a_val = d(LIP["tc"], LIP["bc"])
    b_val = d(LIP["tl"], LIP["bl"])
    c_val = d(LIP["tr"], LIP["br"])
    d_val = d(LIP["cl"], LIP["cr"])
    return 0.0 if d_val < 1e-6 else (a_val + b_val + c_val) / (3 * d_val)


def sts_resample(frames, n_points=150):
    n_frames = len(frames)
    if n_frames == 0:
        return [0.0] * n_points
    if n_frames == 1:
        return [frames[0]] * n_points

    out = [frames[0]]
    for idx in range(1, n_points):
        pos = (n_frames - 1) / (n_points - 1) * idx
        ceil_idx = min(math.ceil(pos), n_frames - 1)
        floor_idx = max(ceil_idx - 1, 0)
        out.append(
            (ceil_idx - pos) * frames[floor_idx]
            + (1 - (ceil_idx - pos)) * frames[ceil_idx]
        )
    return out


@dataclass(frozen=True)
class MarFrameState:
    mar: float = 0.0
    pipe: str = "IDLE"
    rec_dur: float = 0.0
    rec_frames: int = 0
    probs: list[float] = field(default_factory=list)
    pred: int = -1
    yawn_cnt: int = 0
    talk_cnt: int = 0
    yawn_15m_cnt: int = 0
    yawn_alert_on: bool = False
    tcn_ms: float = 0.0
    display_pred: int = -1
    result_seq: int = 0


class MarPipeline:
    def __init__(self, model_path):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise SystemExit("[ERR] pip install onnxruntime") from exc

        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.input_shape = self.sess.get_inputs()[0].shape

        self.roll_buf = deque(maxlen=500)
        self.slope_buf = deque()

        self.pipe = "IDLE"
        self.onset_t0 = None
        self.offset_t0 = None
        self.evt_frames = []
        self.evt_t0 = 0.0
        self.peak_mar = 0.0
        self.rise_t0 = None
        self.mar_prev = 0.0

        self.probs = []
        self.pred = -1
        self.yawn_cnt = 0
        self.talk_cnt = 0
        self.tcn_ms = 0.0
        self.yawn_evt_times = deque()
        self.event_count = 0
        self.display_pred = -1
        self.result_seq = 0

    def _reset_event(self):
        self.evt_frames = []
        self.onset_t0 = None
        self.offset_t0 = None
        self.peak_mar = 0.0

    def _infer(self, now, clock_fn):
        evt_dur_actual = now - self.evt_t0
        if evt_dur_actual < MIN_EVT_DUR:
            self.display_pred = 0
            self._reset_event()
            self.pipe = "MONITORING"
            return

        if evt_dur_actual < STS_THR:
            clip = list(self.evt_frames)
            if len(clip) > 150:
                clip = clip[:150]
            elif len(clip) < 150:
                clip = clip + [clip[-1]] * (150 - len(clip))
        else:
            clip = sts_resample(self.evt_frames, 150)

        x = np.array(clip, dtype=np.float32)[np.newaxis, np.newaxis, :]
        t0_tcn = clock_fn()
        logits = self.sess.run([self.output_name], {self.input_name: x})[0]
        t1_tcn = clock_fn()
        self.tcn_ms = (t1_tcn - t0_tcn) * 1000.0

        exp_l = np.exp(logits - logits.max())
        self.probs = (exp_l / exp_l.sum()).flatten().tolist()
        self.pred = int(np.argmax(logits))
        self.display_pred = self.pred
        self.result_seq += 1

        if self.pred == 2:
            self.yawn_cnt += 1
            self.yawn_evt_times.append(now)
        elif self.pred == 1:
            self.talk_cnt += 1

        self.event_count += 1
        self._reset_event()
        self.pipe = "MONITORING"

    def update(self, face_ok, lms, now, clock=None):
        clock_fn = clock or time.perf_counter

        mar_raw = compute_mar(lms) if face_ok else 0.0

        self.roll_buf.append((now, mar_raw))
        self.slope_buf.append((now, mar_raw))
        while self.slope_buf and now - self.slope_buf[0][0] > SLOPE_WIN_T:
            self.slope_buf.popleft()

        slope_var = (
            float(np.var([value for _, value in self.slope_buf]))
            if len(self.slope_buf) >= 3
            else 0.0
        )

        d_mar = mar_raw - self.mar_prev
        self.mar_prev = mar_raw
        if d_mar > 0:
            if self.rise_t0 is None:
                self.rise_t0 = now
        else:
            self.rise_t0 = None
        rising_dur = (now - self.rise_t0) if self.rise_t0 else 0.0

        if self.pipe == "IDLE":
            if face_ok:
                self.pipe = "MONITORING"

        elif self.pipe == "MONITORING":
            if not face_ok:
                self.pipe = "IDLE"
                self.onset_t0 = None
                self.rise_t0 = None
            elif mar_raw > ONSET_THR and rising_dur >= MIN_RISE_T:
                if self.onset_t0 is None:
                    self.onset_t0 = now
                if now - self.onset_t0 >= ONSET_SUSTAIN_T:
                    self.pipe = "RECORDING"
                    lookback_window = LOOKBACK_T + (now - self.onset_t0)
                    self.evt_frames = [
                        value
                        for ts, value in self.roll_buf
                        if now - ts <= lookback_window
                    ]
                    self.evt_t0 = now
                    self.offset_t0 = None
                    self.peak_mar = max(self.evt_frames) if self.evt_frames else 0.0
            else:
                self.onset_t0 = None

        elif self.pipe == "RECORDING":
            if not face_ok:
                self._reset_event()
                self.pipe = "IDLE"
            else:
                self.evt_frames.append(mar_raw)
                if mar_raw > self.peak_mar:
                    self.peak_mar = mar_raw
                evt_dur = now - self.evt_t0
                if mar_raw < OFFSET_THR and slope_var < SLOPE_VAR_THR:
                    if self.offset_t0 is None:
                        self.offset_t0 = now
                    if now - self.offset_t0 >= OFFSET_SUSTAIN_T:
                        self.pipe = "INFER"
                else:
                    self.offset_t0 = None
                if evt_dur >= MAX_EVT_DUR:
                    if self.peak_mar >= ONSET_THR * 1.5:
                        self.pipe = "INFER"
                    else:
                        self.display_pred = 0
                        self._reset_event()
                        self.pipe = "MONITORING"

        if self.pipe == "INFER":
            self._infer(now, clock_fn)

        while self.yawn_evt_times and now - self.yawn_evt_times[0] > YAWN_ALERT_WIN_S:
            self.yawn_evt_times.popleft()
        yawn_15m_cnt = len(self.yawn_evt_times)
        yawn_alert_on = yawn_15m_cnt >= YAWN_ALERT_COUNT_15M

        rec_dur = now - self.evt_t0 if self.pipe == "RECORDING" else 0.0
        rec_frames = len(self.evt_frames)

        return MarFrameState(
            mar=mar_raw,
            pipe=self.pipe,
            rec_dur=rec_dur,
            rec_frames=rec_frames,
            probs=list(self.probs),
            pred=self.pred,
            yawn_cnt=self.yawn_cnt,
            talk_cnt=self.talk_cnt,
            yawn_15m_cnt=yawn_15m_cnt,
            yawn_alert_on=yawn_alert_on,
            tcn_ms=self.tcn_ms,
            display_pred=self.display_pred,
            result_seq=self.result_seq,
        )
