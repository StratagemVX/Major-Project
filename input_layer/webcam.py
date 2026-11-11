"""Real-time YOLOv8 webcam tracking and motion overlay script."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from perception.detection import (
    compute_approach_velocity,
    compute_centroid,
    compute_velocity,
    resolve_device,
    resolve_weights_path,
    smooth_height,
)


def run_webcam_tracker(camera_index: int = 0, weights_path: Path | None = None) -> None:
    """Run live webcam object tracking with CPU/CUDA fallback and guaranteed camera release."""
    import cv2
    import torch
    from ultralytics import YOLO

    resolved_weights = resolve_weights_path(weights_path)
    if not resolved_weights.is_file():
        raise FileNotFoundError(f"YOLO weights file not found: {resolved_weights}")

    device = resolve_device(bool(torch.cuda.is_available()))
    model = YOLO(str(resolved_weights))
    model.to(device)

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Camera at index {camera_index} cannot be opened")

    previous_positions: dict[int, tuple[int, int, float]] = {}
    previous_heights: dict[int, tuple[float, float]] = {}
    height_smooth: dict[int, float] = {}
    prev_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = frame.copy()
            results = model.track(frame, persist=True, verbose=False)

            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy
                ids = results[0].boxes.id

                for box, obj_id in zip(boxes, ids):
                    track_id = int(obj_id)
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = compute_centroid(x1, y1, x2, y2)
                    height = y2 - y1

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

                    current_time = time.time()
                    vx, vy, speed = compute_velocity(cx, cy, current_time, previous_positions.get(track_id))
                    previous_positions[track_id] = (cx, cy, current_time)

                    smoothed_height = smooth_height(height, height_smooth.get(track_id), alpha=0.7)
                    height_smooth[track_id] = smoothed_height

                    v_approach = compute_approach_velocity(
                        smoothed_height,
                        current_time,
                        previous_heights.get(track_id),
                    )
                    previous_heights[track_id] = (smoothed_height, current_time)

                    scale = 0.1
                    end_x = int(cx + vx * scale)
                    end_y = int(cy + vy * scale)
                    cv2.arrowedLine(annotated_frame, (cx, cy), (end_x, end_y), (255, 0, 0), 2)
                    cv2.putText(
                        annotated_frame,
                        f"ID {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        annotated_frame,
                        f"V:{speed:.1f}",
                        (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 255),
                        2,
                    )
                    if abs(v_approach) > 1:
                        cv2.putText(
                            annotated_frame,
                            f"A:{v_approach:.1f}",
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            2,
                        )

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if prev_time != 0.0 else 0.0
            prev_time = curr_time

            cv2.putText(
                annotated_frame,
                f"FPS:{int(fps)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("YOLO Motion Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_tracker()
