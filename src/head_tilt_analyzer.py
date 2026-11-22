import math
import utils

_NOSE_TIP, _EYE_L, _EYE_R = 1, 33, 263

NEUTRAL_PITCH = None   # None = auto-calibrate from the first CAL_FRAMES frames. Or hardcode your number.
CAL_FRAMES = 45        # ~1.5 s at 30 fps: keep your head level and look at the screen
DOWN_RANGE = 0.25      # (raw pitch looking fully down) - neutral
UP_RANGE = 0.20        # neutral - (raw pitch looking fully up)
DEADZONE = 0.30        # fraction of the range treated as neutral
MAX_YAW = 0.45         # skip frames where the head is turned too far sideways
EMA_ALPHA = 0.3        # smoothing: lower = steadier but laggier

_ema = None
_cal = {"sum": 0.0, "n": 0, "neutral": None}


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def reset_head_pitch():
    global _ema
    _ema = None


def recalibrate_head_pitch():
    """Call when the person is sitting level (e.g. on a key press). Restarts the neutral measurement."""
    _cal.update(sum=0.0, n=0, neutral=None)


def head_pitch(face_landmarks):
    """Raw pitch proxy: bigger = looking down, smaller = looking up. None if unusable."""
    if not face_landmarks or len(face_landmarks) <= _EYE_R:
        return None

    nose = face_landmarks[_NOSE_TIP]
    eye_l = face_landmarks[_EYE_L]
    eye_r = face_landmarks[_EYE_R]

    eye_dist = math.hypot(eye_r.x - eye_l.x, eye_r.y - eye_l.y)
    if eye_dist < 1e-6:
        return None

    mid_x = (eye_l.x + eye_r.x) / 2.0
    mid_y = (eye_l.y + eye_r.y) / 2.0

    if abs(nose.x - mid_x) / eye_dist > MAX_YAW:   # head turned sideways
        return None

    return (nose.y - mid_y) / eye_dist


def head_pitch_value(face_landmarks):
    """Signed pitch in [-1, 1]. -1 = up, 0 = neutral, +1 = down. None if unusable or still calibrating."""
    global _ema

    raw = head_pitch(face_landmarks)
    if raw is None:
        _ema = None
        return None

    _ema = raw if _ema is None else EMA_ALPHA * raw + (1 - EMA_ALPHA) * _ema

    # Neutral: hardcoded value if you set one, otherwise measured at startup
    neutral = NEUTRAL_PITCH if NEUTRAL_PITCH is not None else _cal["neutral"]
    if neutral is None:
        _cal["sum"] += _ema
        _cal["n"] += 1
        if _cal["n"] >= CAL_FRAMES:
            _cal["neutral"] = _cal["sum"] / _cal["n"]
            print(f"[head pitch] neutral calibrated: {_cal['neutral']:.3f}")
        return 0.0   # report neutral while calibrating

    off = _ema - neutral
    value = _clamp(off / DOWN_RANGE if off >= 0 else off / UP_RANGE, -1.0, 1.0)

    mag = abs(value)
    if mag <= DEADZONE:
        return 0.0
    mag = (mag - DEADZONE) / (1.0 - DEADZONE)
    return mag if value > 0 else -mag


def head_pitch_label(value):
    """Classification from the signed value. None if the head pose is unusable."""
    if value is None:
        return None
    if value == 0.0:
        return utils.HeadTilt.NEUTRAL
    down = value > 0
    if abs(value) < 0.35:
        return utils.HeadTilt.SLIGHT_DOWN if down else utils.HeadTilt.SLIGHT_UP
    return utils.HeadTilt.DOWN if down else utils.HeadTilt.UP