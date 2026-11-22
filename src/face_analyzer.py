import stabilizer
import utils

BodyLanguageMode = utils.BodyLanguageMode

def get_face_confidence_scores(face_blendshapes):
    if not face_blendshapes:
        return []

    face_confidence_scores = []

    # ^ ============================================ SMILE
    smile_left = face_blendshapes[44].score
    smile_right = face_blendshapes[45].score
    smile_confidence = ((smile_left + smile_right) / 2) 


    # ^ ============================================ CLOSED EYES
    eyeSquintLeft = face_blendshapes[19].score
    eyeSquintRight = face_blendshapes[20].score
    raw_average = (eyeSquintLeft + eyeSquintRight) / 2
    balance_factor = min(eyeSquintLeft, eyeSquintRight) / (max(eyeSquintLeft, eyeSquintRight) + 1e-6)  # Find how close they are to being equally closed (0.0 to 1.0)
    closed_eyes_confidence = raw_average * balance_factor


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


    # ^ ============================================ SAD
    brow_inner_up = face_blendshapes[3].score
    mouth_frown_left = face_blendshapes[36].score
    mouth_frown_right = face_blendshapes[37].score
    sad_confidence = (brow_inner_up > 0.93) / 2 + mouth_frown_left + mouth_frown_right


    
    # ^ ============================================ RETURN VALUES
    face_confidence_scores = [
        {"mode": BodyLanguageMode.SMILE, "score": smile_confidence},
        {"mode": BodyLanguageMode.CLOSED_EYES, "score": closed_eyes_confidence},
        {"mode": BodyLanguageMode.SIDE_EYE, "score": side_eye_confidence},
        {"mode": BodyLanguageMode.SUSPICIOUS, "score": suspicious_confidence},
        {"mode": BodyLanguageMode.ANGRY, "score": angry_confidence},
        {"mode": BodyLanguageMode.SAD, "score": sad_confidence},
    ]
    
    return face_confidence_scores

