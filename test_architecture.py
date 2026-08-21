import numpy as np
import os
import sys
import tempfile
import cv2

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from config import setup_logger
from src.business_logic.models import FaceProfile, DetectionResult
from src.business_logic.face_manager import FaceManager
from src.business_logic.face_recognizer import FaceRecognizer
from src.business_logic.activity_logger import ActivityLogger, ActivityLogEntry
from src.utils.sound_player import SoundService

logger = setup_logger("test_runner")

def test_models():
    print("Testing models...")
    profile = FaceProfile(name="Test User", encodings=[np.zeros(128)])
    assert profile.name == "Test User"
    assert profile.primary_encoding is not None
    assert len(profile.encodings) == 1

    det = DetectionResult(
        name="Test User",
        confidence=95.5,
        distance=0.25,
        location=(10, 100, 100, 10),
        is_known=True
    )
    assert det.is_known is True
    assert det.bounding_box == (10, 100, 100, 10)
    print("[PASSED] Models test.")

def test_activity_logger_and_csv():
    print("Testing ActivityLogger & CSV export...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        alerts_dir = os.path.join(tmp_dir, "alerts")
        snapshots_dir = os.path.join(tmp_dir, "snapshots")
        csv_file = os.path.join(tmp_dir, "test_log.csv")

        act_logger = ActivityLogger(alerts_dir=alerts_dir, snapshots_dir=snapshots_dir, log_cooldown=0.0)

        # Log known detection
        det_known = DetectionResult(name="Alice", confidence=92.0, distance=0.2, location=(0, 10, 10, 0), is_known=True)
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        e1 = act_logger.log_detection(det_known, dummy_frame)
        assert e1 is not None
        assert e1.is_known is True

        # Log unknown detection (should auto-snapshot)
        det_unknown = DetectionResult(name="Unknown", confidence=0.0, distance=0.8, location=(0, 10, 10, 0), is_known=False)
        e2 = act_logger.log_detection(det_unknown, dummy_frame)
        assert e2 is not None
        assert e2.is_known is False
        assert e2.snapshot_path is not None
        assert os.path.exists(e2.snapshot_path)

        # Manual snapshot
        success_snap, snap_path = act_logger.save_manual_snapshot(dummy_frame)
        assert success_snap is True
        assert os.path.exists(snap_path)

        # CSV Export
        success_csv, msg = act_logger.export_to_csv(csv_file)
        assert success_csv is True
        assert os.path.exists(csv_file)
        with open(csv_file, "r") as f:
            content = f.read()
            assert "Alice" in content
            assert "Unknown" in content

    print("[PASSED] ActivityLogger & CSV export test.")

def test_face_recognizer_hud():
    print("Testing FaceRecognizer & HUD...")
    recognizer = FaceRecognizer(tolerance=0.55)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = DetectionResult(name="Eliav", confidence=95.0, distance=0.2, location=(40, 140, 140, 40), is_known=True)
    annotated = recognizer.draw_annotations(blank_frame, [det], fps=30.5, show_confidence=True)
    assert annotated.shape == blank_frame.shape
    print("[PASSED] FaceRecognizer & HUD test.")

if __name__ == "__main__":
    test_models()
    test_activity_logger_and_csv()
    test_face_recognizer_hud()
    print("\nALL PERFORMANCE, LOGGING & HUD TESTS PASSED SUCCESSFULLY!")
