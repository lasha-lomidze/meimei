import time
import cv2
import numpy as np
from PIL import Image, ImageSequence
from mapper import mapper


# Define your fixed window dimensions (Width, Height)
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600

loaded_image_mapper = {i: None for i in mapper}


def _load_gif_frames(image_path):
    frames = []
    durations = []
    with Image.open(image_path) as gif:
        for frame in ImageSequence.Iterator(gif):
            rgb = np.array(frame.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            frames.append(cv2.resize(bgr, (WINDOW_WIDTH, WINDOW_HEIGHT)))
            durations.append(max(int(frame.info.get("duration", 100)), 20) / 1000.0)
    return frames, durations


# load the images
for i in mapper:
    image_path = mapper[i]
    if image_path.lower().endswith(".gif"):
        try:
            frames, durations = _load_gif_frames(image_path)
            if frames:
                loaded_image_mapper[i] = {"frames": frames, "durations": durations}
                continue
        except Exception:
            pass

    loaded_image = cv2.imread(image_path)
    if loaded_image is None:
        print(f"Error: Could not load '{image_path}'. Creating a gray placeholder instead.")
        image = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8) * 128
    else:
        image = cv2.resize(loaded_image, (WINDOW_WIDTH, WINDOW_HEIGHT))
    loaded_image_mapper[i] = image

# Initialize the window explicitly before the loop
blank_window = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
cv2.namedWindow("meimei", cv2.WINDOW_NORMAL)
cv2.resizeWindow("meimei", WINDOW_WIDTH, WINDOW_HEIGHT)

_active_gif_mode = None
_gif_start_time = 0.0

def show_window(image):
    cv2.imshow("meimei", image)

def display(body_language_mode):
    global _active_gif_mode, _gif_start_time

    asset = loaded_image_mapper.get(body_language_mode)
    if isinstance(asset, dict):
        if _active_gif_mode != body_language_mode:
            _active_gif_mode = body_language_mode
            _gif_start_time = time.monotonic()

        frames = asset["frames"]
        durations = asset["durations"]
        total_duration = sum(durations) or 1.0
        elapsed = (time.monotonic() - _gif_start_time) % total_duration

        acc = 0.0
        frame_idx = 0
        for idx, dur in enumerate(durations):
            acc += dur
            if elapsed < acc:
                frame_idx = idx
                break
        show_window(frames[frame_idx])
    elif asset is not None:
        _active_gif_mode = None
        show_window(asset)
    else:
        _active_gif_mode = None
        show_window(blank_window)






