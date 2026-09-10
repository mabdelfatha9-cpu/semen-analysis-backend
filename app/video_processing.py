"""Extract a sample of frames from an uploaded video for analysis."""

from typing import List, Tuple

import cv2
import numpy as np


def extract_frames(video_path: str, max_frames: int = 30) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        frames = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % 5 == 0:
                frames.append(frame)
            idx += 1
            if len(frames) >= max_frames:
                break
        cap.release()
        return frames

    indices = np.linspace(0, total_frames - 1, num=min(max_frames, total_frames), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)

    cap.release()
    return frames


def frame_dimensions(frame: np.ndarray) -> Tuple[int, int]:
    height, width = frame.shape[:2]
    return width, height


def extract_sequential_frames(video_path: str, max_frames: int = 60) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1.0:
        fps = 30.0

    frames: List[np.ndarray] = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)

    cap.release()
    return frames, fps
