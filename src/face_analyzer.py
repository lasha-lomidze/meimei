import head_tilt_analyzer
import utils
import math

BodyLanguageMode = utils.BodyLanguageMode
HeadTilt = utils.HeadTilt
_base = {"w": None, "n": 0}

def get_face_confidence_scores(face_blendshapes, face_landmarks):
    if not face_blendshapes:
        return []

    face_confidence_scores = []
    v = head_tilt_analyzer.head_pitch_value(face_landmarks)   # e.g. 0.42
    label = head_tilt_analyzer.head_pitch_label(v)  # e.g. HeadTilt.SLIGHT_DOWN

    # ^ ============================================ SMILE
    d = lambda a, b: math.hypot(face_landmarks[a].x - face_landmarks[b].x, face_landmarks[a].y - face_landmarks[b].y)
    w = d(61, 291) / d(234, 454)                      # mouth width / face width
    if _base["n"] < 60:                               # first ~2 s: learn your neutral width (keep a neutral face)
        _base["w"] = w if _base["w"] is None else (_base["w"] * _base["n"] + w) / (_base["n"] + 1)
        _base["n"] += 1
    smile_confidence = min(max((w / _base["w"] - 1.01) / 0.10, 0.0), 1.0)
    


    # ^ ============================================ CLOSED EYES
    eyeSquintLeft = face_blendshapes[19].score
    eyeSquintRight = face_blendshapes[20].score
    raw_average = (eyeSquintLeft + eyeSquintRight) / 2
    balance_factor = min(eyeSquintLeft, eyeSquintRight) / (max(eyeSquintLeft, eyeSquintRight) + 1e-6)  # Find how close they are to being equally closed (0.0 to 1.0)
    closed_eyes_confidence = (raw_average * balance_factor) 


    # ^ ============================================ SIDE EYE
    eye_look_in_left = face_blendshapes[13].score
    eye_look_out_right = face_blendshapes[16].score
    side_eye_confidence = (eye_look_out_right + eye_look_in_left) / 2.0


    # ^ ============================================ SUSPICIOUS
    bs = {c.category_name: c.score for c in face_blendshapes}
    base = {"browOuterUpRight": 0.0, "browOuterUpLeft": 0.0,
        "browDownLeft": 0.0, "browDownRight": 0.0}

    up_diff   = (bs["browOuterUpRight"] - base["browOuterUpRight"]) - (bs["browOuterUpLeft"] - base["browOuterUpLeft"])
    down_diff = (bs["browDownLeft"]     - base["browDownLeft"])     - (bs["browDownRight"]    - base["browDownRight"])

    right_brow_signal = (max(up_diff, 0.0) + max(down_diff, 0.0)) / 2.0
    left_brow_signal = (max(-up_diff, 0.0) + max(-down_diff, 0.0)) / 2.0
    suspicious_confidence = max(right_brow_signal, left_brow_signal) 
    if label != HeadTilt.NEUTRAL: suspicious_confidence *= 1.2



    # ^ ============================================ ANGRY
    brow_down_left = face_blendshapes[2].score
    brow_down_right = face_blendshapes[1].score
    cheek_puff = face_blendshapes[6].score
    nose_sneer_left = face_blendshapes[50].score
    nose_sneer_right = face_blendshapes[51].score
    angry_confidence = (
        brow_down_left * 2
        + brow_down_right * 2
        + cheek_puff * 3
        + nose_sneer_left
        + nose_sneer_right
    ) * 1.7
    if label == HeadTilt.SLIGHT_DOWN or label == HeadTilt.DOWN: angry_confidence *= 1.3
    if label == HeadTilt.SLIGHT_UP or label == HeadTilt.UP: angry_confidence *= 0.9


    # ^ ============================================ SAD
    brow_inner_up = face_blendshapes[3].score
    mouth_frown_left = face_blendshapes[36].score
    mouth_frown_right = face_blendshapes[37].score
    sad_confidence = (brow_inner_up > 0.91) / 2 + mouth_frown_left + mouth_frown_right
    if label == HeadTilt.SLIGHT_UP: sad_confidence *= 1.2
    if label == HeadTilt.UP: sad_confidence *= 1.1
    if label == HeadTilt.SLIGHT_DOWN: sad_confidence *= 0.9
    if label == HeadTilt.DOWN: sad_confidence *= 0.8


    
    # ^ ============================================ RETURN VALUES
    face_confidence_scores = [
        {"mode": BodyLanguageMode.SMILE, "score": smile_confidence},
        {"mode": BodyLanguageMode.CLOSED_EYES, "score": closed_eyes_confidence},
        {"mode": BodyLanguageMode.SIDE_EYE, "score": side_eye_confidence},
        {"mode": BodyLanguageMode.SUSPICIOUS, "score": suspicious_confidence},
        {"mode": BodyLanguageMode.ANGRY, "score": angry_confidence},
        {"mode": BodyLanguageMode.SAD, "score": sad_confidence},
    ]

    print(face_confidence_scores)
    return face_confidence_scores

