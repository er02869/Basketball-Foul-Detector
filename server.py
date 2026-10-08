"""Standalone CourtVision web application server."""

from __future__ import annotations

import tempfile
from pathlib import Path
from time import perf_counter

from flask import Flask, jsonify, request, send_from_directory

from courtvision_core import (
    DETECTORS,
    MAX_FRAMES,
    TRAVEL_INFERENCE_FRAME_OPTIONS,
    VIDEO_EXTENSIONS,
    predict_with_timing,
    sample_video_frames,
)


WEB_DIR = Path(__file__).resolve().parent / "web"
#serve the browser files and API from the same local application
app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/api/status")
def status():
    #show which saved violation models are available to run
    return jsonify({"detectors": [{"name": detector.name, "description": detector.description, "available": detector.model_path is not None and detector.model_path.is_file()} for detector in DETECTORS]})


@app.post("/api/analyze")
def analyze():
    upload = request.files.get("video")
    if upload is None or not upload.filename:
        return jsonify({"error": "Choose a video file first."}), 400

    #reject unsupported formats before passing the clip to OpenCV
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        return jsonify({"error": "Unsupported video format."}), 400

    #validate the travel-only frame count selected in the browser
    try:
        travel_frame_limit = int(request.form.get("travel_frames", str(MAX_FRAMES)))
    except (TypeError, ValueError):
        return jsonify({"error": "Choose 8, 16, or 32 travel frames."}), 400
    if travel_frame_limit not in TRAVEL_INFERENCE_FRAME_OPTIONS:
        return jsonify({"error": "Choose 8, 16, or 32 travel frames."}), 400

    try:
        #OpenCV needs a filesystem path, so save each upload temporarily
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temporary_file:
            upload.save(temporary_file.name)
            #measure decoding separately from preprocessing and inference
            decoding_started = perf_counter()
            frames = sample_video_frames(Path(temporary_file.name))
            decoding_ms = (perf_counter() - decoding_started) * 1000
            results = []
            #use the same sampled frames to run every available detector
            for detector in DETECTORS:
                result = {"name": detector.name, "description": detector.description}
                try:
                    #apply the selected smaller batch only to the travel detector
                    frame_limit = travel_frame_limit if detector.inference_frame_options else None
                    label, confidence, timings = predict_with_timing(detector, frames, frame_limit)
                    result.update(label=label, confidence=confidence, available=True, timings_ms=timings)
                except FileNotFoundError:
                    result.update(error="No trained model is available yet.", available=False)
                except (OSError, ValueError, RuntimeError) as error:
                    result.update(error=f"Unable to run detector: {error}", available=False)
                results.append(result)
    except (OSError, ValueError) as error:
        return jsonify({"error": str(error)}), 422

    return jsonify({
        "filename": upload.filename,
        "frame_count": len(frames),
        "decoding_ms": decoding_ms,
        "travel_frame_limit": travel_frame_limit,
        "results": results,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8502, debug=False)
