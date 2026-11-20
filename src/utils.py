import enum
import cv2
from screeninfo import get_monitors

class BodyLanguageMode(enum.Enum):
    OPEN_PALM = "Open Palm"
    FIST = "Closed Fist"
    SMILE = "Smile"
    CLOSED_EYES = "Closed Eyes"
    MIDDLE_FINGER = "Middle Finger"
    SIDE_EYE = "Side Eye"
    SUSPICIOUS = "Suspicious"
    TWO_OPEN_PALMS = "Two Open Palms"
    ANGRY = "Angry"
    SAD = "Sad"
    SECRET = "Secret"
    TWO_PEOPLE = "Two People"
    MULTIPLE_PEOPLE = "Multiple People"
    DANCING = "Dancing"
    UNKNOWN = "Unknown Gesture"


prev_scores = [[] for _ in range(15)]


def get_screen_resolution(fallback=(1920, 1080)):
    try:
        monitors = get_monitors()
        m = next((m for m in monitors if m.is_primary), monitors[0])
        return m.width, m.height
    except Exception:
        return fallback

def center_windows():
    # this constants are hard coded here, not ideal generally but convenient for now
    MEIMEI_WINDOW_NAME = "meimei"
    WEBCAM_WINDOW_NAME = "Webcam"
    WINDOW_SIZE = 600

    screen_width, screen_height = get_screen_resolution()

    cv2.moveWindow(MEIMEI_WINDOW_NAME, (screen_width - WINDOW_SIZE) // 2 - (WINDOW_SIZE // 2), (screen_height - round(WINDOW_SIZE * 1.5)) // 2)
    cv2.moveWindow(WEBCAM_WINDOW_NAME, (screen_width - WINDOW_SIZE) // 2 + (WINDOW_SIZE // 2), (screen_height - round(WINDOW_SIZE * 1.5)) // 2)
