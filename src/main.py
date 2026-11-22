import mediapipe as mp
import cv2
import os
import sys
import zipfile
import hand_analyzer
import meimei
import face_analyzer
import pose_analyzer
import utils
import stabilizer



# =====================================================================
# * General Configuration
# =====================================================================



BodyLanguageMode = utils.BodyLanguageMode 
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600
stabilizer_instance = stabilizer.Stabilizer()
prev_scores = stabilizer_instance.prev_scores



# =====================================================================
# * Media pipe Configuration
# =====================================================================



def resource_path(relative_path: str) -> str: # Get the absolute path to a resource, without it you cant import .task
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)


BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="tasks/holistic_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    min_face_detection_confidence=0.1,
    min_pose_detection_confidence=0.5,
    output_face_blendshapes=True
)

landmarker = HolisticLandmarker.create_from_options(options)

with zipfile.ZipFile("tasks/holistic_landmarker.task") as zf:
    face_detector_model_bytes = zf.read("face_detector.tflite")

face_detector_options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_buffer=face_detector_model_bytes),
    running_mode=VisionRunningMode.VIDEO,
    min_detection_confidence=0.5,
)
face_detector = FaceDetector.create_from_options(face_detector_options)



# =====================================================================
# * Main
# =====================================================================



prev_time = 0
frame_timestamp_ms = 0
cap = cv2.VideoCapture(0)
cv2.namedWindow("Webcam")
utils.center_windows()


if not cap.isOpened():
    print("Error: Could not open video capture.")
    sys.exit()


try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1) # flip the camera because it is inverted
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert the frame to RGB for Mediapipe

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_timestamp_ms += 33

        face_result = face_detector.detect_for_video(mp_image, frame_timestamp_ms)
        if len(face_result.detections) == 2: 
            pose_analyzer.reset_motion_history()
            meimei.display(BodyLanguageMode.TWO_PEOPLE)
        if len(face_result.detections) > 2:
            pose_analyzer.reset_motion_history()
            meimei.display(BodyLanguageMode.MULTIPLE_PEOPLE)
        else:
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            gesture_scores = hand_analyzer.get_hand_confidence_scores(result.left_hand_landmarks)
            expression_scores = face_analyzer.get_face_confidence_scores(result.face_blendshapes, result.face_landmarks)
            pose_scores = pose_analyzer.get_pose_confidence_scores(result.pose_landmarks)

            # detech if there are two open palms
            if result.left_hand_landmarks and result.right_hand_landmarks:
                left_hand_score = hand_analyzer.get_hand_confidence_scores(result.left_hand_landmarks)
                right_hand_score = hand_analyzer.get_hand_confidence_scores(result.right_hand_landmarks)
                if left_hand_score and right_hand_score:
                    # Only consider it as TWO_OPEN_PALMS if both hands have a high confidence for OPEN_PALM
                    if left_hand_score[0]["mode"] == BodyLanguageMode.OPEN_PALM and right_hand_score[0]["mode"] == BodyLanguageMode.OPEN_PALM:
                        # make OpenaPalm (single hand) score 0.0 so it doesnt interfere
                        gesture_scores = [score for score in gesture_scores if score["mode"] != BodyLanguageMode.OPEN_PALM]
                        gesture_scores.append({"mode": BodyLanguageMode.TWO_OPEN_PALMS, "score": min(left_hand_score[0]["score"], right_hand_score[0]["score"])})

            stabilizer_instance.append_scores([*gesture_scores, *expression_scores, *pose_scores])
            final_mode = stabilizer_instance.get_stable_mode()
        
            meimei.display(final_mode)

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
finally:
    landmarker.close()
    face_detector.close()
    cap.release()
    cv2.destroyAllWindows()


