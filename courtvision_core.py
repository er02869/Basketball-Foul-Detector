"""Shared capstone video sampling and model inference."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
import cv2
import numpy as np
import tensorflow as tf


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = (224, 224)
MAX_FRAMES = 32
#offer smaller inference batches only for the travel classifier
TRAVEL_INFERENCE_FRAME_OPTIONS = (8, 16, 32)
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".wmv"}


@dataclass(frozen=True)
class Detector:
    name: str
    description: str
    model_path: Path | None
    positive_label: str
    negative_label: str
    #empty options keep other detectors on the full shared frame sample
    inference_frame_options: tuple[int, ...] = ()


#connect each violation type to its saved training model and output labels
DETECTORS = (
    Detector(
        "Foul",
        "Contact or illegal defensive action",
        next((path for path in (PROJECT_DIR / "FOUL-DETECTOR/BEST-RUNS/basketball_foul_model.keras", PROJECT_DIR / "FOUL-DETECTOR/BEST-RUNS/basketball_foul_model(1).keras") if path.is_file()), None),
        "Foul",
        "Clean",
    ),
    Detector(
        "Travel",
        "Illegal steps or movement with the ball",
        PROJECT_DIR / "TRAVEL-DETECTOR/BEST-RUNS/basketball_travel_model.keras",
        "Travel",
        "Clean",
        TRAVEL_INFERENCE_FRAME_OPTIONS,
    ),
    Detector("Flagrant foul", "Excessive or unnecessary contact", PROJECT_DIR / "FLAGRANT-FOULS/BEST-RUNS/basketball_flagrant_model.keras", "Flagrant foul", "Clean"),
    Detector("Goaltending", "Illegal interference with a shot", PROJECT_DIR / "GOAL-TENDING/BEST-RUNS/basketball_goaltending_model.keras", "Goaltending", "Clean"),
)


def sample_video_frames(video_path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("OpenCV could not open this video.")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    if frame_count > 0:
        #sample evenly across the clip instead of processing every frame
        indexes = np.linspace(0, frame_count - 1, min(frame_count, MAX_FRAMES), dtype=int)
        for index in indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            success, frame = capture.read()
            if success:
                frames.append(frame)
    else:
        #fall back to reading sequentially when video metadata is unavailable
        while len(frames) < MAX_FRAMES:
            success, frame = capture.read()
            if not success:
                break
            frames.append(frame)
    capture.release()

    if not frames:
        raise ValueError("No readable frames were found in this video.")
    return frames


#keep large models in memory between uploads instead of reloading each time
_model_cache: dict[str, Any] = {}


def load_model(model_path: str) -> Any:
    if model_path not in _model_cache:
        _model_cache[model_path] = tf.keras.models.load_model(model_path, compile=False)
    return _model_cache[model_path]


def select_inference_frames(
    detector: Detector,
    frames: list[np.ndarray],
    frame_limit: int | None = None,
) -> list[np.ndarray]:
    if not frames:
        raise ValueError("No video frames were provided for inference.")
    if frame_limit is None:
        return frames
    if frame_limit not in detector.inference_frame_options:
        raise ValueError("This detector does not support the requested frame count.")
    if frame_limit >= len(frames):
        return frames

    #select evenly across the decoded frames to preserve coverage of the clip
    indexes = np.linspace(0, len(frames) - 1, frame_limit, dtype=int)
    return [frames[int(index)] for index in indexes]


def predict_with_timing(
    detector: Detector,
    frames: list[np.ndarray],
    frame_limit: int | None = None,
) -> tuple[str, float, dict[str, float | int]]:
    if detector.model_path is None or not detector.model_path.is_file():
        raise FileNotFoundError("No saved model is available yet.")

    selected_frames = select_inference_frames(detector, frames, frame_limit)
    model_started = perf_counter()
    model = load_model(str(detector.model_path))
    model_load_ms = (perf_counter() - model_started) * 1000

    #prepare frames in the same size, color order, and type used for inference
    preprocessing_started = perf_counter()
    batch = np.stack([cv2.cvtColor(cv2.resize(frame, IMAGE_SIZE), cv2.COLOR_BGR2RGB) for frame in selected_frames]).astype("float32")
    preprocessing_ms = (perf_counter() - preprocessing_started) * 1000

    inference_started = perf_counter()
    scores = np.asarray(model.predict(batch, verbose=0)).reshape(-1)
    inference_ms = (perf_counter() - inference_started) * 1000

    #average frame scores, then map the binary score to a detector label
    score = float(np.mean(scores))
    label, confidence = (detector.positive_label, score) if score >= 0.5 else (detector.negative_label, 1.0 - score)
    timings = {
        "model_load_ms": model_load_ms,
        "preprocessing_ms": preprocessing_ms,
        "inference_ms": inference_ms,
        "processed_frames": len(selected_frames),
    }
    return label, confidence, timings


def predict(detector: Detector, frames: list[np.ndarray]) -> tuple[str, float]:
    label, confidence, _ = predict_with_timing(detector, frames)
    return label, confidence
