from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np

RIGHT_EYE = {
    "tl": 160,
    "bl": 144,
    "tr": 158,
    "br": 153,
    "cl": 133,
    "cr": 33,
}
LEFT_EYE = {
    "tl": 385,
    "bl": 373,
    "tr": 387,
    "br": 380,
    "cl": 362,
    "cr": 263,
}

SMOOTH_TAU_S = 0.020
RAW_MEDIAN_LEN = 3
USE_RAW_EAR_FOR_STATE = False

REF_WINDOW_S = 30.0
READY_MIN_SAMPLES = 30
OPEN_REF_Q = 0.90
CLOSE_REF_Q = 0.01

OPEN_REF_UP_TAU_S = 0.50
OPEN_REF_DOWN_TAU_S = 3.00
CLOSE_REF_TAU_S = 5.00

CLOSED_ENTER_PCT = 25.0
CLOSED_EXIT_PCT = 35.0
CLOSED_ENTER_HOLD_S = 0.000
CLOSED_EXIT_HOLD_S = 0.000
CLOSED_ENTER_MEAN_PCT = 20.0
CLOSED_EXIT_MEAN_PCT = 35.0
CLOSED_ENTER_MIN_EYE_PCT = 11.0
CLOSED_EXIT_MIN_EYE_PCT = 35.0

MIN_SPAN_ABS = 0.005
MIN_SPAN_RATIO = 0.02
MAX_OPEN_PCT = 100.0

BILATERAL_GAP_BRIDGE_S = 0.000
MIN_BLINK_S = 0.000
MAX_BLINK_S = 0.500
DROWSY_ALERT_S = 2.0
FACE_LOST_RESET_S = 2.0


def time_alpha(dt: float, tau_s: float) -> float:
    if tau_s <= 1e-9:
        return 1.0
    if dt <= 0.0:
        return 1.0
    return 1.0 - math.exp(-dt / tau_s)


def ema_time(prev: float | None, cur: float, dt: float, tau_s: float) -> float:
    if prev is None:
        return cur
    alpha = time_alpha(dt, tau_s)
    return prev + alpha * (cur - prev)


class TimedValueBuffer:
    def __init__(self, window_s: float):
        self.window_s = float(window_s)
        self.buf: Deque[tuple[float, float]] = deque()

    def trim(self, now: float) -> None:
        while self.buf and (now - self.buf[0][0]) > self.window_s:
            self.buf.popleft()

    def push(self, now: float, value: float) -> None:
        self.buf.append((float(now), float(value)))
        self.trim(now)

    def values(self, now: float | None = None) -> list[float]:
        if now is not None:
            self.trim(now)
        return [value for _, value in self.buf]

    def percentile(
        self,
        q: float,
        now: float | None = None,
        default: float | None = None,
    ) -> float | None:
        vals = self.values(now)
        if not vals:
            return default
        q_pct = 100.0 * q if q <= 1.0 else q
        return float(np.percentile(np.asarray(vals, dtype=np.float32), q_pct))

    def __len__(self) -> int:
        return len(self.buf)


def compute_ear(lms, eye) -> float:
    def dy(a: str, b: str) -> float:
        return abs(lms[eye[a]].y - lms[eye[b]].y)

    def d3(a: str, b: str) -> float:
        idx_a, idx_b = eye[a], eye[b]
        dx = lms[idx_a].x - lms[idx_b].x
        dy_val = lms[idx_a].y - lms[idx_b].y
        dz = lms[idx_a].z - lms[idx_b].z
        return math.sqrt(dx * dx + dy_val * dy_val + dz * dz)

    a_val = dy("tl", "bl")
    b_val = dy("tr", "br")
    d_val = d3("cl", "cr")
    return 0.0 if d_val < 1e-6 else (a_val + b_val) / (2.0 * d_val)


@dataclass(frozen=True)
class EyeSideState:
    raw: float = 0.0
    smooth: float = 0.0
    open_ref: float | None = None
    close_ref: float | None = None
    open_pct: float = 0.0
    state: str = "NO_FACE"
    ready: bool = False
    samples: int = 0
    span: float = 0.0


@dataclass(frozen=True)
class EyeFrameState:
    face_ok: bool = False
    ready: bool = False
    left: EyeSideState = field(default_factory=EyeSideState)
    right: EyeSideState = field(default_factory=EyeSideState)
    overall_state: str = "NO_FACE"
    bilateral_closed_run_s: float = 0.0
    max_bilateral_closed_run_s: float = 0.0
    alert_on: bool = False
    asym: float = 0.0
    face_lost_s: float = 0.0


@dataclass(frozen=True)
class EyeEvent:
    start_s: float
    end_s: float
    duration_s: float
    kind: str
    is_blink: bool
    alert_2s: bool


def classify_event_duration(duration_s: float) -> tuple[str, bool, bool]:
    alert_2s = duration_s >= DROWSY_ALERT_S
    if alert_2s:
        return "DROWSY", False, True
    if duration_s < MIN_BLINK_S:
        return "SHORT", False, False
    if duration_s > MAX_BLINK_S:
        return "LONG", False, False
    return "BLINK", True, False


class EyeModel:
    def __init__(self, name: str):
        self.name = name
        self.raw_med_buf: Deque[float] = deque(maxlen=RAW_MEDIAN_LEN)
        self.ref_buf = TimedValueBuffer(REF_WINDOW_S)

        self.prev_t: float | None = None
        self.smooth: float | None = None
        self.curr_raw = 0.0
        self.curr_smooth = 0.0
        self.curr_dt = 0.0

        self.open_ref: float | None = None
        self.close_ref: float | None = None
        self.open_pct = 100.0
        self.state = "OPEN"

        self.closed = False
        self.closed_enter_t0: float | None = None
        self.closed_exit_t0: float | None = None
        self.sample_count = 0

    def reset(self) -> None:
        self.__init__(self.name)

    def observe(self, raw: float, now: float) -> float:
        if self.prev_t is None:
            dt = 1.0 / 30.0
        else:
            dt = max(now - self.prev_t, 1e-6)

        self.raw_med_buf.append(float(raw))
        med = float(np.median(np.asarray(self.raw_med_buf, dtype=np.float32)))
        self.smooth = ema_time(self.smooth, med, dt, SMOOTH_TAU_S)

        self.curr_raw = float(raw)
        self.curr_smooth = float(self.smooth)
        self.curr_dt = float(dt)
        self.prev_t = float(now)
        return self.curr_smooth

    def _span_floor(self, open_ref: float) -> float:
        return max(MIN_SPAN_ABS, MIN_SPAN_RATIO * max(open_ref, 1e-6))

    def _normalized_open_pct(
        self,
        x_val: float,
        open_ref: float,
        close_ref: float,
    ) -> tuple[float, float]:
        span_floor = self._span_floor(open_ref)
        eff_close = min(close_ref, open_ref - span_floor)
        span = max(open_ref - eff_close, span_floor)
        pct = 100.0 * (x_val - eff_close) / span
        return float(np.clip(pct, 0.0, MAX_OPEN_PCT)), float(span)

    def commit(self, now: float) -> EyeSideState:
        x_val = self.curr_raw if USE_RAW_EAR_FOR_STATE else self.curr_smooth
        raw = self.curr_raw
        dt = self.curr_dt

        self.ref_buf.push(now, x_val)
        self.sample_count += 1

        cand_open = self.ref_buf.percentile(OPEN_REF_Q, now=now, default=x_val)
        cand_close = self.ref_buf.percentile(CLOSE_REF_Q, now=now, default=x_val)

        if self.open_ref is None:
            self.open_ref = cand_open
        else:
            tau = OPEN_REF_UP_TAU_S if cand_open >= self.open_ref else OPEN_REF_DOWN_TAU_S
            self.open_ref = ema_time(self.open_ref, cand_open, dt, tau)

        if self.close_ref is None:
            self.close_ref = cand_open * 0.3
        else:
            self.close_ref = ema_time(self.close_ref, cand_close, dt, CLOSE_REF_TAU_S)

        if self.open_ref is None:
            self.open_ref = x_val
        if self.close_ref is None:
            self.close_ref = x_val

        max_close = self.open_ref - self._span_floor(self.open_ref)
        if self.close_ref > max_close:
            self.close_ref = max_close

        self.open_pct, span = self._normalized_open_pct(x_val, self.open_ref, self.close_ref)

        if not self.closed:
            if self.open_pct <= CLOSED_ENTER_PCT:
                if self.closed_enter_t0 is None:
                    self.closed_enter_t0 = now
                if (now - self.closed_enter_t0) >= CLOSED_ENTER_HOLD_S:
                    self.closed = True
                    self.closed_exit_t0 = None
            else:
                self.closed_enter_t0 = None
        else:
            if self.open_pct >= CLOSED_EXIT_PCT:
                if self.closed_exit_t0 is None:
                    self.closed_exit_t0 = now
                if (now - self.closed_exit_t0) >= CLOSED_EXIT_HOLD_S:
                    self.closed = False
                    self.closed_enter_t0 = None
            else:
                self.closed_exit_t0 = None

        self.state = "CLOSED" if self.closed else "OPEN"

        ready = self.sample_count >= READY_MIN_SAMPLES
        return EyeSideState(
            raw=float(raw),
            smooth=float(x_val),
            open_ref=float(self.open_ref),
            close_ref=float(self.close_ref),
            open_pct=float(self.open_pct),
            state=self.state,
            ready=ready,
            samples=int(self.sample_count),
            span=float(span),
        )


class EyeTracker:
    def __init__(self):
        self.left = EyeModel("L")
        self.right = EyeModel("R")

        self.face_lost_t0: float | None = None

        self.bilateral_closed_since: float | None = None
        self.last_bilateral_closed_t: float | None = None
        self.max_bilateral_closed_run_s = 0.0
        self.alert_on = False

        self.events: list[EyeEvent] = []

    def _close_current_event_if_needed(self) -> EyeEvent | None:
        event = None
        if self.bilateral_closed_since is not None and self.last_bilateral_closed_t is not None:
            dur = max(0.0, self.last_bilateral_closed_t - self.bilateral_closed_since)
            kind, is_blink, alert_2s = classify_event_duration(dur)
            event = EyeEvent(
                start_s=self.bilateral_closed_since,
                end_s=self.last_bilateral_closed_t,
                duration_s=dur,
                kind=kind,
                is_blink=is_blink,
                alert_2s=alert_2s,
            )
            self.events.append(event)
        self.bilateral_closed_since = None
        self.last_bilateral_closed_t = None
        self.alert_on = False
        return event

    def _face_lost_duration(self, now: float) -> float:
        if self.face_lost_t0 is None:
            return 0.0
        return max(0.0, now - self.face_lost_t0)

    def _make_noface_state(self, now: float) -> EyeFrameState:
        left_ready = self.left.sample_count >= READY_MIN_SAMPLES
        right_ready = self.right.sample_count >= READY_MIN_SAMPLES
        ready = left_ready and right_ready

        left_state = EyeSideState(
            raw=0.0,
            smooth=float(self.left.smooth or 0.0),
            open_ref=self.left.open_ref,
            close_ref=self.left.close_ref,
            open_pct=float(self.left.open_pct if self.left.smooth is not None else 0.0),
            state="NO_FACE",
            ready=left_ready,
            samples=self.left.sample_count,
            span=float(max(0.0, (self.left.open_ref or 0.0) - (self.left.close_ref or 0.0))),
        )
        right_state = EyeSideState(
            raw=0.0,
            smooth=float(self.right.smooth or 0.0),
            open_ref=self.right.open_ref,
            close_ref=self.right.close_ref,
            open_pct=float(self.right.open_pct if self.right.smooth is not None else 0.0),
            state="NO_FACE",
            ready=right_ready,
            samples=self.right.sample_count,
            span=float(max(0.0, (self.right.open_ref or 0.0) - (self.right.close_ref or 0.0))),
        )

        if self.bilateral_closed_since is not None:
            run_s = max(0.0, now - self.bilateral_closed_since)
            self.max_bilateral_closed_run_s = max(self.max_bilateral_closed_run_s, run_s)
            self.alert_on = run_s >= DROWSY_ALERT_S
        else:
            run_s = 0.0
            self.alert_on = False

        return EyeFrameState(
            face_ok=False,
            ready=ready,
            left=left_state,
            right=right_state,
            overall_state="NO_FACE",
            bilateral_closed_run_s=run_s,
            max_bilateral_closed_run_s=self.max_bilateral_closed_run_s,
            alert_on=self.alert_on,
            asym=0.0,
            face_lost_s=self._face_lost_duration(now),
        )

    def update(self, face_ok: bool, lms, now: float) -> EyeFrameState:
        valid = bool(face_ok) and lms is not None

        if not valid:
            if self.face_lost_t0 is None:
                self.face_lost_t0 = now

            if self.bilateral_closed_since is not None and self.last_bilateral_closed_t is not None:
                if (now - self.last_bilateral_closed_t) > BILATERAL_GAP_BRIDGE_S:
                    self._close_current_event_if_needed()

            if self._face_lost_duration(now) > FACE_LOST_RESET_S:
                self.left.reset()
                self.right.reset()
                self.face_lost_t0 = now

            return self._make_noface_state(now)

        ear_l = compute_ear(lms, LEFT_EYE)
        ear_r = compute_ear(lms, RIGHT_EYE)
        if not (np.isfinite(ear_l) and np.isfinite(ear_r)):
            return self._make_noface_state(now)

        self.face_lost_t0 = None

        left_smooth = self.left.observe(float(ear_l), now)
        right_smooth = self.right.observe(float(ear_r), now)

        left_state = self.left.commit(now)
        right_state = self.right.commit(now)

        state_both_closed = left_state.state == "CLOSED" and right_state.state == "CLOSED"
        mean_open_pct = 0.5 * (left_state.open_pct + right_state.open_pct)
        min_eye_open_pct = min(left_state.open_pct, right_state.open_pct)
        if self.bilateral_closed_since is None:
            soft_closed = (
                mean_open_pct <= CLOSED_ENTER_MEAN_PCT
                and min_eye_open_pct <= CLOSED_ENTER_MIN_EYE_PCT
            )
        else:
            soft_closed = (
                mean_open_pct <= CLOSED_EXIT_MEAN_PCT
                and min_eye_open_pct <= CLOSED_EXIT_MIN_EYE_PCT
            )
        both_closed = state_both_closed or soft_closed

        overall_state = "BOTH_CLOSED" if both_closed else "OPEN"

        if both_closed:
            if self.bilateral_closed_since is None:
                self.bilateral_closed_since = now
            self.last_bilateral_closed_t = now
        elif self.bilateral_closed_since is not None and self.last_bilateral_closed_t is not None:
            if (now - self.last_bilateral_closed_t) > BILATERAL_GAP_BRIDGE_S:
                self._close_current_event_if_needed()

        if self.bilateral_closed_since is not None:
            run_s = max(0.0, now - self.bilateral_closed_since)
            self.max_bilateral_closed_run_s = max(self.max_bilateral_closed_run_s, run_s)
            self.alert_on = run_s >= DROWSY_ALERT_S
        else:
            run_s = 0.0
            self.alert_on = False

        ready = left_state.ready and right_state.ready
        return EyeFrameState(
            face_ok=True,
            ready=ready,
            left=left_state,
            right=right_state,
            overall_state=overall_state,
            bilateral_closed_run_s=run_s,
            max_bilateral_closed_run_s=self.max_bilateral_closed_run_s,
            alert_on=self.alert_on,
            asym=abs(left_smooth - right_smooth),
            face_lost_s=0.0,
        )

    def finalize(self, now: float) -> None:
        if self.bilateral_closed_since is not None and self.last_bilateral_closed_t is not None:
            self.last_bilateral_closed_t = now
        self._close_current_event_if_needed()
