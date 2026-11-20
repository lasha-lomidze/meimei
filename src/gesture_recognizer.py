import math
import utils

BodyLanguageMode = utils.BodyLanguageMode


def get_gesture_confidence_scores(hand_landmarks):
    if not hand_landmarks:
        return []

    gesture_confidence_scores = []


    # ^ ============================================ FINGER EXTENSION
    wrist = hand_landmarks[0]
    finger_tips = [8, 12, 16, 20]
    finger_bases = [5, 9, 13, 17]

    finger_scores = []
    for tip, base in zip(finger_tips, finger_bases):
        dist_tip = math.sqrt(
            (hand_landmarks[tip].x - wrist.x) ** 2
            + (hand_landmarks[tip].y - wrist.y) ** 2
        )
        dist_base = math.sqrt(
            (hand_landmarks[base].x - wrist.x) ** 2
            + (hand_landmarks[base].y - wrist.y) ** 2
        )
        extension_ratio = dist_tip / (dist_base + 1e-6)
        finger_score = max(0.0, min(1.0, (extension_ratio - 1.0) / 0.7))
        finger_scores.append(finger_score)

    index_score, middle_score, ring_score, pinky_score = finger_scores



    # ^ ============================================ THUMB EXTENSION
    thumb_tip = hand_landmarks[4]
    index_base = hand_landmarks[5]
    thumb_base = hand_landmarks[2]

    dist_thumb_to_index = math.sqrt(
        (thumb_tip.x - index_base.x) ** 2
        + (thumb_tip.y - index_base.y) ** 2
    )
    dist_thumb_base = math.sqrt(
        (thumb_base.x - wrist.x) ** 2
        + (thumb_base.y - wrist.y) ** 2
    )
    thumb_ratio = dist_thumb_to_index / (dist_thumb_base + 1e-6)
    thumb_score = max(0.0, min(1.0, (thumb_ratio - 0.4) / 0.8))



    # ^ ============================================ OPEN PALM
    average_finger_score = sum(finger_scores) / 4.0
    open_palm_confidence = (average_finger_score * 0.8) + (thumb_score * 0.2)



    # ^ ============================================ FIST
    average_curled_finger_score = sum(1.0 - score for score in finger_scores) / 4.0
    fist_confidence = (average_curled_finger_score * 0.8) + ((1.0 - thumb_score) * 0.2)



    # ^ ============================================ MIDDLE FINGER
    other_fingers_curled_average = (
        (1.0 - index_score) + (1.0 - ring_score) + (1.0 - pinky_score)
    ) / 3.0
    middle_finger_confidence = (middle_score * 0.7) + (other_fingers_curled_average * 0.3)



    # ^ ============================================ RETURN VALUES
    gesture_confidence_scores = [
        {"gesture": BodyLanguageMode.OPEN_PALM, "score": open_palm_confidence},
        {"gesture": BodyLanguageMode.FIST, "score": fist_confidence},
        {"gesture": BodyLanguageMode.MIDDLE_FINGER, "score": middle_finger_confidence},
    ]

    return gesture_confidence_scores
