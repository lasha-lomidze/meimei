import math
import utils

_NOSE_TIP, _EYE_L, _EYE_R = 1, 33, 263

# Calibrate these from YOUR camera setup. Defaults are guesses.
NEUTRAL_PITCH = 0.50   # raw pitch when your head is level
DOWN_RANGE = 0.25      # (raw pitch looking fully down) - NEUTRAL_PITCH
UP_RANGE = 0.20        # NEUTRAL_PITCH - (raw pitch looking fully up)
DEADZONE = 0.08        # |value| below this is reported as exactly 0
MAX_YAW = 0.45         # skip frames where the head is turned too far sideways


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


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

    # Head turned sideways -> the estimate is unreliable
    if abs(nose.x - mid_x) / eye_dist > MAX_YAW:
        return None

    return (nose.y - mid_y) / eye_dist


def head_pitch_value(face_landmarks):
    """Signed pitch in [-1, 1]. -1 = fully up, 0 = neutral, +1 = fully down. None if unusable."""
    raw = head_pitch(face_landmarks)
    if raw is None:
        return None

    off = raw - NEUTRAL_PITCH
    # Separate ranges because looking up and looking down don't move the ratio equally
    value = off / DOWN_RANGE if off >= 0 else off / UP_RANGE
    value = _clamp(value, -1.0, 1.0)

    # Deadzone: rescale so the output ramps smoothly from 0 at the edge of the zone
    mag = abs(value)
    if mag <= DEADZONE:
        return 0.0
    mag = (mag - DEADZONE) / (1.0 - DEADZONE)
    return mag if value > 0 else -mag


def head_pitch_label(value):
    """Finer classification from the signed value."""
    if value is None:
        return None
    mag = abs(value)
    if mag == 0.0:
        return utils.HeadTilt.NEUTRAL
    if mag < 0.35:
        return utils.HeadTilt.SLIGHT_DOWN if value > 0 else utils.HeadTilt.SLIGHT_UP
    if mag < 0.70:
        return utils.HeadTilt.DOWN if value > 0 else utils.HeadTilt.UP
    return utils.HeadTilt.DOWN if value > 0 else utils.HeadTilt.UP