# meimei 🐾

A real-time computer vision and expression/pose classifier that mirrors your emotions, hand gestures, and body language using reactive reactions and memes.

Built using **MediaPipe**, **OpenCV**, and custom heuristic stabilization algorithms, **meimei** captures your webcam feed, tracks facial blendshapes, hand landmarks, and full-body pose kinematics, and maps them to dynamic visual responses in real time.

---

## ✨ Features

- **Multi-Modal Detection**:
  - **Facial Blendshapes & Head Tilt Analysis**: Evaluates eyebrow furrowing, eye squints, smiles, frowns, gaze direction, and pitch angle (head tilts) to distinguish nuanced emotions like anger vs. sadness.
  - **Hand Gesture Recognition**: Detects finger articulation, palm orientation, and gesture states (e.g., Open Palm, Closed Fist, Middle Finger).
  - **Pose & Motion Dynamics**: Tracks upper-body kinematics and repetitive landmark displacement over time to detect actions like dancing.
  - **Multi-Person Face Counting**: Distinguishes between solo interactions, pairs (`TWO_PEOPLE`), and crowd situations (`MULTIPLE_PEOPLE`).
- **Custom Frame Stabilizer**:
  - Employs a sliding-window confidence-weighted smoothing algorithm to eliminate frame jitter, flickering, and momentary false positives.
  - Smooths transitions across 15 distinct body language & gesture states.
- **Dual-Window Interface**:
  - Automatically queries screen resolution to center and align the live webcam feed alongside the reactive meimei display.

---

## 🎭 Supported Modes & Expressions

`meimei` recognizes **15 distinct states**:

| Mode | Trigger |
| :--- | :--- |
| **Smile** | Raised mouth corners with cheek squint |
| **Sad** | Mouth frown + inner eyebrow raise / subtle head tilt |
| **Angry** | Eyebrow furrow / brow down with downward head tilt |
| **Side Eye** | Lateral gaze deviation without full head turn |
| **Suspicious** | Eyebrow furrow + squinted eyes |
| **Closed Eyes** | Extended eye blink / closed eyelids |
| **Open Palm** | Single hand open towards the camera |
| **Two Open Palms** | Both hands open simultaneously |
| **Closed Fist** | Clenched fist gesture |
| **Middle Finger** | Middle finger extended, other digits curled |
| **Dancing** | Rhythmic periodic arm/wrist movement across consecutive frames |
| **Two People** | Exactly two faces detected in frame |
| **Multiple People**| Three or more faces detected in frame |
| **Unknown Gesture**| Neutral / unclassified posture |
| **Secret** | Easter egg expression combination |

---

## 🛠️ Tech Stack & Requirements

- **Python 3.9+**
- **[OpenCV](https://opencv.org/)** (`opencv-python`): Real-time frame capture, rendering, and window management.
- **[MediaPipe](https://developers.google.com/mediapipe)**: Holistic landmarker (face blendshapes, hand landmarks, pose kinematics) and face detection tasks.
- **[screeninfo](https://github.com/rr-/screeninfo)**: Dynamic monitor resolution detection for window positioning.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/lasha-lomidze/meimei.git
cd meimei
```

### 2. Set Up a Virtual Environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Task Model

Ensure `tasks/holistic_landmarker.task` is in place:
```
meimei/
├── tasks/
│   └── holistic_landmarker.task
```

### 5. Run Application

```bash
python3 src/main.py
```

*Press `q` at any time while focused on the Webcam window to exit.*

---

## 📂 Project Structure

```text
meimei/
├── public/                 # Reaction image assets & gifs
│   ├── images/             # Standard reaction assets for each mode
│   └── secret/             # Easter egg visual assets
├── src/
│   ├── face_analyzer.py    # Blendshape parsing & facial emotion heuristics
│   ├── hand_analyzer.py    # Finger articulation & gesture calculation
│   ├── head_tilt_analyzer.py# Pitch / roll / yaw angle estimation
│   ├── pose_analyzer.py    # Motion history buffer & dancing pose heuristics
│   ├── stabilizer.py       # Multi-frame confidence stabilizer
│   ├── mapper.py           # Maps BodyLanguageMode enum to image paths
│   ├── meimei.py           # Window renderer for reactive images
│   ├── utils.py            # Enums, monitor helpers & window positioning
│   └── main.py             # Video capture loop & holistic pipeline
├── tasks/                  # MediaPipe holistic task model bundle
├── requirements.txt        # Python package dependencies
├── LICENSE                 # MIT License
└── README.md
```

---

## 📝 License

Distributed under the [MIT License](LICENSE).
