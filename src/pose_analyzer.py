from collections import deque, namedtuple
import math
import utils

BodyLanguageMode = utils.BodyLanguageMode

_MOTION_HISTORY_SIZE = 18
_JITTER_DEADZONE = 0.012
_TELEPORT_THRESHOLD = 0.6
_MIN_VIS = 0.6  # minimum landmark visibility for the SECRET check
_motion_history = deque(maxlen=_MOTION_HISTORY_SIZE)

_Point = namedtuple("_Point", "x y")


def reset_motion_history():
    _motion_history.clear()


def _dist_2d(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _clamp(value, min_val=0.0, max_val=1.0):
    return max(min_val, min(max_val, value))


def _ramp(value, lo, hi):
    """0 at lo, 1 at hi, linear in between."""
    return _clamp((value - lo) / (hi - lo))


def _visible(*pts):
    return all(getattr(p, "visibility", 1.0) >= _MIN_VIS for p in pts)


def _elbow_angle_deg(shoulder, elbow, wrist):
    v1x, v1y = shoulder.x - elbow.x, shoulder.y - elbow.y
    v2x, v2y = wrist.x - elbow.x, wrist.y - elbow.y
    mag1 = math.sqrt(v1x ** 2 + v1y ** 2)
    mag2 = math.sqrt(v2x ** 2 + v2y ** 2)
    cos_angle = (v1x * v2x + v1y * v2y) / (mag1 * mag2 + 1e-6)
    return math.degrees(math.acos(_clamp(cos_angle, -1.0, 1.0)))


def _hand_to_head_score(pose_landmarks, shoulder_idx, elbow_idx, wrist_idx, pinky_idx, index_idx, head_scale, left_ear, right_ear, head_top):
    shoulder = pose_landmarks[shoulder_idx]
    elbow = pose_landmarks[elbow_idx]
    wrist = pose_landmarks[wrist_idx]
    pinky = pose_landmarks[pinky_idx]
    index = pose_landmarks[index_idx]

    # Occluded or guessed arm landmarks -> no score
    if not _visible(shoulder, elbow, wrist):
        return 0.0

    if _visible(pinky, index):
        hand_mid = _Point((pinky.x + index.x) / 2.0, (pinky.y + index.y) / 2.0)
    else:
        hand_mid = wrist

    # 1. Proximity of wrist / hand to ears or top of head
    #    Full score at <= 0.6 head-widths, zero at >= 1.0
    min_head_dist = min(
        _dist_2d(wrist, left_ear),
        _dist_2d(wrist, right_ear),
        _dist_2d(hand_mid, left_ear),
        _dist_2d(hand_mid, right_ear),
        _dist_2d(hand_mid, head_top),
        _dist_2d(wrist, head_top),
    )
    proximity_score = 1.0 - _ramp(min_head_dist / head_scale, 0.6, 1.0)

    # 2. Elevation: hand must reach ear level or above
    #    Starts at 0.85, full at 1.05 (ratio 1.0 = ear height)
    ear_y = (left_ear.y + right_ear.y) / 2.0
    hand_y = min(wrist.y, hand_mid.y)
    elevation_ratio = (shoulder.y - hand_y) / (abs(shoulder.y - ear_y) + 1e-4)
    elevation_score = _ramp(elevation_ratio, 0.85, 1.05)

    # 3. Elbow bend: hand behind/on head requires a bent elbow
    angle_deg = _elbow_angle_deg(shoulder, elbow, wrist)
    bend_score = _clamp((135.0 - angle_deg) / 55.0)

    # min instead of geometric mean: one weak factor can't be hidden by the others
    return min(proximity_score, elevation_score, bend_score)


def _dancing_score(pose_landmarks, shoulder_dist):
    nose = pose_landmarks[0]
    left_shoulder = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]

    mid_x = (left_shoulder.x + right_shoulder.x) / 2.0
    mid_y = (left_shoulder.y + right_shoulder.y) / 2.0
    tilt_y = left_shoulder.y - right_shoulder.y

    sample = (mid_x, mid_y, tilt_y, nose.x, nose.y, left_shoulder.x, left_shoulder.y, right_shoulder.x, right_shoulder.y)

    if _motion_history:
        prev = _motion_history[-1]
        jump = math.sqrt((sample[0] - prev[0]) ** 2 + (sample[1] - prev[1]) ** 2) / shoulder_dist
        if jump > _TELEPORT_THRESHOLD:
            _motion_history.clear()

    _motion_history.append(sample)

    if len(_motion_history) < 5:
        return 0.0

    # Shoulder side-to-side wiggle & tilt oscillation
    total_x_path = 0.0
    total_tilt_path = 0.0
    total_body_path = 0.0

    for i in range(1, len(_motion_history)):
        prev = _motion_history[i - 1]
        curr = _motion_history[i]

        dx = abs(curr[0] - prev[0]) / shoulder_dist
        dtilt = abs(curr[2] - prev[2]) / shoulder_dist

        body_step = (
            math.sqrt((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2)
            + math.sqrt((curr[3] - prev[3]) ** 2 + (curr[4] - prev[4]) ** 2)
            + math.sqrt((curr[5] - prev[5]) ** 2 + (curr[6] - prev[6]) ** 2)
            + math.sqrt((curr[7] - prev[7]) ** 2 + (curr[8] - prev[8]) ** 2)
        ) / (4.0 * shoulder_dist)

        total_x_path += max(0.0, dx - _JITTER_DEADZONE)
        total_tilt_path += max(0.0, dtilt - _JITTER_DEADZONE)
        total_body_path += max(0.0, body_step - _JITTER_DEADZONE)

    # Normalize by window fill so score scales consistently as history fills
    window_scale = (_MOTION_HISTORY_SIZE - 1) / max(len(_motion_history) - 1, 1)
    scaled_x_path = total_x_path * window_scale
    scaled_tilt_path = total_tilt_path * window_scale
    scaled_body_path = total_body_path * window_scale

    wiggle_score = _clamp((scaled_x_path * 1.4 + scaled_tilt_path * 1.6) / 0.85)
    body_motion_score = _clamp(scaled_body_path / 0.75)

    return max(wiggle_score, body_motion_score)


def get_pose_confidence_scores(pose_landmarks):
    if not pose_landmarks or len(pose_landmarks) < 21:
        reset_motion_history()
        return []

    left_ear = pose_landmarks[7]
    right_ear = pose_landmarks[8]
    left_shoulder = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]

    ear_dist = _dist_2d(left_ear, right_ear)
    shoulder_dist = _dist_2d(left_shoulder, right_shoulder)
    head_scale = max(ear_dist, shoulder_dist * 0.4, 1e-4)

    head_top = _Point(
        (left_ear.x + right_ear.x) / 2.0,
        min(left_ear.y, right_ear.y) - 0.35 * head_scale,
    )

    # ^ ============================================ SECRET
    # Skip if either ear is occluded/guessed (e.g. head turned): head_scale and head_top become unreliable
    if _visible(left_ear, right_ear):
        left_arm_score = _hand_to_head_score(
            pose_landmarks, 11, 13, 15, 17, 19, head_scale, left_ear, right_ear, head_top
        )
        right_arm_score = _hand_to_head_score(
            pose_landmarks, 12, 14, 16, 18, 20, head_scale, left_ear, right_ear, head_top
        )
        secret_confidence = max(left_arm_score, right_arm_score)
    else:
        secret_confidence = 0.0

    # ^ ============================================ DANCING
    dancing_confidence = _dancing_score(pose_landmarks, max(shoulder_dist, 1e-4))

    # ^ ============================================ RETURN VALUES
    pose_confidence_scores = [
        {"mode": BodyLanguageMode.SECRET, "score": secret_confidence},
        {"mode": BodyLanguageMode.DANCING, "score": dancing_confidence},
    ]

    return pose_confidence_scores