import os
import time
import csv
import cv2
import uuid
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from config import setup_logger
from src.business_logic.models import DetectionResult

logger = setup_logger(__name__)

@dataclass
class ActivityLogEntry:
    """
    Represents a recognized visitor or security event log entry.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    time_str: str = ""
    name: str = "Unknown"
    confidence: float = 0.0
    is_known: bool = False
    snapshot_path: Optional[str] = None

    def __post_init__(self):
        if not self.time_str:
            self.time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))


class ActivityLogger:
    """
    Manages real-time visitor attendance logs, CSV export, and auto-snapshot alerts.
    """
    def __init__(
        self,
        alerts_dir: str = "alerts",
        snapshots_dir: str = "snapshots",
        log_cooldown: float = 5.0,
        cooldown: Optional[float] = None
    ):
        self.alerts_dir = alerts_dir
        self.snapshots_dir = snapshots_dir
        self.log_cooldown = cooldown if cooldown is not None else log_cooldown
        self.entries: List[ActivityLogEntry] = []
        self.last_log_times = {}

        # Ensure storage folders exist
        os.makedirs(self.alerts_dir, exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)

    def log_detection(self, detection: DetectionResult, frame: Optional[np.ndarray] = None) -> Optional[ActivityLogEntry]:
        """
        Record a detection event with debounce cooldown and auto-snapshot unknown visitors.
        
        :param detection: DetectionResult object
        :param frame: Optional BGR frame to snapshot
        :return: ActivityLogEntry if recorded, None if suppressed by cooldown
        """
        current_time = time.time()
        key = detection.name if detection.is_known else "Unknown"
        last_time = self.last_log_times.get(key, 0.0)

        if current_time - last_time < self.log_cooldown:
            return None

        self.last_log_times[key] = current_time

        snapshot_path = None

        # Auto-snapshot unknown visitor
        if not detection.is_known and frame is not None and frame.size > 0:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            filename = f"unknown_{timestamp_str}_{str(uuid.uuid4())[:4]}.jpg"
            snapshot_path = os.path.join(self.alerts_dir, filename)
            try:
                cv2.imwrite(snapshot_path, frame)
                logger.info(f"Auto-saved alert snapshot for unknown visitor: {snapshot_path}")
            except Exception as e:
                logger.error(f"Failed to save alert snapshot: {e}")
                snapshot_path = None

        entry = ActivityLogEntry(
            timestamp=current_time,
            name=detection.name,
            confidence=detection.confidence,
            is_known=detection.is_known,
            snapshot_path=snapshot_path
        )

        self.entries.insert(0, entry)  # Prepend newest entry first
        # Limit memory history to 500 entries
        if len(self.entries) > 500:
            self.entries.pop()

        return entry

    def save_manual_snapshot(self, frame: np.ndarray) -> Tuple[bool, str]:
        """
        Save a full-resolution manual snapshot from live camera.
        """
        if frame is None or frame.size == 0:
            return False, "No active frame to capture."

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp_str}.jpg"
        filepath = os.path.join(self.snapshots_dir, filename)

        try:
            cv2.imwrite(filepath, frame)
            logger.info(f"Saved manual snapshot: {filepath}")
            return True, filepath
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")
            return False, f"Failed to save snapshot: {str(e)}"

    def export_to_csv(self, filepath: str = "attendance_log.csv") -> Tuple[bool, str]:
        """
        Export all recorded activity entries to a CSV file.
        """
        if not self.entries:
            return False, "No activity records to export."

        try:
            with open(filepath, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Header
                writer.writerow(["Timestamp", "Person Name", "Status", "Confidence (%)", "Alert Snapshot File"])
                for entry in self.entries:
                    status = "Recognized" if entry.is_known else "Unknown"
                    writer.writerow([
                        entry.time_str,
                        entry.name,
                        status,
                        f"{entry.confidence:.1f}%" if entry.is_known else "N/A",
                        entry.snapshot_path or ""
                    ])

            logger.info(f"Exported {len(self.entries)} record(s) to {filepath}")
            return True, f"Successfully exported {len(self.entries)} records to '{filepath}'"
        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")
            return False, f"Export failed: {str(e)}"

    def get_entries(self) -> List[ActivityLogEntry]:
        """Return all log entries."""
        return self.entries

    def clear_entries(self) -> None:
        """Clear recorded log entries."""
        self.entries.clear()
        self.last_log_times.clear()
