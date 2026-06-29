"""
Quick smoke-test for the Tracking Pipeline.

Usage:
    # Run on a video file (processes first 300 frames by default):
    python test_tracking.py match.mp4

    # Run on RTSP stream:
    python test_tracking.py rtsp://camera.local:554/stream

    # Process every other frame (faster):
    python test_tracking.py match.mp4 --skip 1

    # Save annotated output video:
    python test_tracking.py match.mp4 --save output.mp4

    # Process only N frames:
    python test_tracking.py match.mp4 --frames 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
from loguru import logger

# ---------------------------------------------------------------------------
# Make sure the package is importable even without `pip install -e .`
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from fie.tracking.calibration import IdentityCalibration, ManualCalibration
from fie.tracking.detector import FootballDetector
from fie.tracking.pipeline import TrackingFrame, TrackingPipeline
from fie.tracking.source import FrameData, VideoFileSource, make_source


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FIE Tracking smoke test")
    p.add_argument("video", help="Path to video file or rtsp:// URL")
    p.add_argument("--skip", type=int, default=0, help="Skip N frames between each processed frame")
    p.add_argument("--frames", type=int, default=300, help="Max frames to process (0 = all)")
    p.add_argument("--start", type=int, default=0, help="Start from this frame number in the video")
    p.add_argument("--save", type=str, default="", help="Save annotated video to this path")
    p.add_argument("--json", type=str, default="", help="Save tracking JSON to this path")
    p.add_argument("--device", type=str, default="cuda", help="cuda / cpu / mps")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=== FIE Tracking Test ===")
    logger.info("Video  : {}", args.video)
    logger.info("Device : {}", args.device)
    logger.info("Skip   : {}", args.skip)
    logger.info("Start  : frame {}", args.start)
    logger.info("Max frames: {}", args.frames or "all")

    # Override device from CLI
    import fie.config as cfg_module
    cfg_module.settings.yolo_device = args.device

    # --- Detector & Pipeline ---
    detector = FootballDetector(device=args.device)
    # Using IdentityCalibration — coordinates will be in pixels, not metres.
    # To get metres, provide ManualCalibration with real pitch corner points.
    pipeline = TrackingPipeline(
        calibration=IdentityCalibration(),
        detector=detector,
        log_interval=50,
    )

    # --- Video source ---
    try:
        source = make_source(args.video, skip_frames=args.skip)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error("Cannot open video: {}", e)
        sys.exit(1)

    # --- Seek to start frame ---
    if args.start > 0:
        logger.info("Seeking to frame {}...", args.start)
        cap = source._cap if hasattr(source, "_cap") else None
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
            logger.info("Seeked OK")

    # --- Video writer (optional) ---
    writer = None
    if args.save:
        w, h = source.resolution
        fps = source.fps / (args.skip + 1)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
        logger.info("Saving annotated video to: {}", args.save)

    # --- Run ---
    all_frames: list[dict] = []
    t0 = time.monotonic()
    frame_count = 0

    with source:
        # We need raw frames for annotation, so we iterate the source manually
        # and call pipeline.process_frame() one by one.
        for raw_frame_data in source:
            if args.frames and frame_count >= args.frames:
                break

            tracking_frame = pipeline.process_frame(raw_frame_data)
            all_frames.append(tracking_frame.to_dict())

            # Print summary every 25 frames
            if frame_count % 25 == 0:
                n_players = len(tracking_frame.players)
                ball = tracking_frame.ball
                ball_str = f"ball=({ball.x:.0f},{ball.y:.0f})" if ball else "ball=None"
                logger.info(
                    "Frame {:>5} | t={:.2f}s | {} players | {}",
                    tracking_frame.frame_idx,
                    tracking_frame.timestamp,
                    n_players,
                    ball_str,
                )

            # Annotate and write (optional)
            if writer is not None:
                det_result = detector.detect(
                    raw_frame_data.frame,
                    raw_frame_data.frame_idx,
                    raw_frame_data.timestamp,
                )
                annotated = detector.annotate(raw_frame_data.frame, det_result)
                # Draw field coords on frame
                for player in tracking_frame.players:
                    cv2.putText(
                        annotated,
                        f"#{player.player_id} {player.speed:.1f}km/h",
                        (10, 30 + player.player_id * 20 % 400),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                writer.write(annotated)

            frame_count += 1

    elapsed = time.monotonic() - t0
    fps_actual = frame_count / elapsed if elapsed > 0 else 0

    # --- Summary ---
    logger.info("")
    logger.info("=== RESULTS ===")
    logger.info("Processed : {} frames in {:.1f}s ({:.1f} fps)", frame_count, elapsed, fps_actual)

    if all_frames:
        last = all_frames[-1]
        logger.info("Last frame: {} players, ball={}", len(last["players"]), last["ball"])

        # Speed stats across all frames
        all_speeds = [
            p["speed"]
            for f in all_frames
            for p in f["players"]
            if p["speed"] > 0
        ]
        if all_speeds:
            logger.info(
                "Speed stats: min={:.1f} avg={:.1f} max={:.1f} km/h",
                min(all_speeds),
                sum(all_speeds) / len(all_speeds),
                max(all_speeds),
            )

    # Print first frame as JSON for inspection
    if all_frames:
        logger.info("")
        logger.info("Sample frame (frame 0):")
        print(json.dumps(all_frames[0], indent=2))

    # --- Save JSON (optional) ---
    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_frames, f, indent=2)
        logger.info("Tracking data saved to: {}", args.json)

    # --- Cleanup ---
    if writer:
        writer.release()
        logger.info("Annotated video saved to: {}", args.save)


if __name__ == "__main__":
    main()
