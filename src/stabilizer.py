import utils
from collections import defaultdict

BodyLanguageMode = utils.BodyLanguageMode


class Stabilizer:
    def __init__(self):
        self.MAX_PREV_SCORES_TO_TRACK = 15
        self.MAX_PREV_FRAMES_TO_TRACK = 15
        self.CONFIDENCE_THRESHOLD = 0.4
        self.SUSPICIOUS_CONFIDENCE_THRESHOLD = 0.012
        self.prev_scores = []
        self.N = 10
        self.X = 6

    def append_scores(self, scores):
        if len(self.prev_scores) >= self.MAX_PREV_SCORES_TO_TRACK:
            self.prev_scores.pop(0)
        self.prev_scores.append(scores)

    def get_highest_confidence_element(self, frame):
        if not frame:
            return None
        return max(frame, key=lambda x: x["score"])

    def eliminate_by_threshold(self):
        for i in range(len(self.prev_scores)):
            self.prev_scores[i] = [score for score in self.prev_scores[i] if score["score"] >= self.CONFIDENCE_THRESHOLD]

    def eliminate_by_consistency(self):
        for i in range(1, len(self.prev_scores)):
            current_frame = self.prev_scores[i]
            previous_frame = self.prev_scores[i - 1]
            current_highest = max(current_frame, key=lambda x: x["score"])["mode"] if current_frame else None
            previous_highest = max(previous_frame, key=lambda x: x["score"])["mode"] if previous_frame else None
            if current_highest and previous_highest and current_highest != previous_highest:
                self.prev_scores[i] = [score for score in current_frame if score["mode"] == previous_highest]


    def get_stable_mode(self):
        unknown = BodyLanguageMode.UNKNOWN
        window = list(self.prev_scores)[-self.N:]          # last n frames
        tops = [t for t in map(self.get_highest_confidence_element, window) if t]

        by_mode = defaultdict(list)
        for t in tops:
            by_mode[t["mode"]].append(t["score"])
        if not by_mode:
            return unknown

        mode, scores = max(by_mode.items(), key=lambda kv: len(kv[1]))
        if len(scores) < self.X:                            # not enough agreeing frames
            return unknown

        threshold = (
            self.SUSPICIOUS_CONFIDENCE_THRESHOLD
            if mode is BodyLanguageMode.SUSPICIOUS
            else self.CONFIDENCE_THRESHOLD
        )
        return mode if sum(scores) / len(scores) >= threshold else unknown

    # check n previous frames take out max confidence element from each frame, if from those x or more are same (x being less then n) then take average confidence of those frames and see if it passes threshold




# * prev_scores_look = [[{"mode": BodyLanguageMode.UNKNOWN, "score": 0.0}, {"mode": BodyLanguageMode.OPEN_PALM, "score": 0.8}, ...], [...], ...]


# stabilizer by:
# eliminate by threshold
# stabilize by checking the consistency of the highest confidence mode across the last N frames
# check how close top confidence scores are to each other to determine stability

# add head tilt to certain expressions so they are easier to detect
# for two people or 3 people make sure they are in webcam for at least 5-10 frames before showing according photo
# some expressions are only confident when im actively making them, and they are harder to keep stable over multiple frames, so consider adding additional checks or longer observation windows to confirm them before displaying, im thinking we ease the checks if we detect sudden spike or short history of them showing couple ms/sc ago
# this could involve dynamically adjusting the confidence threshold or the number of agreeing frames required based on recent history to better capture transient expressions without compromising overall stability