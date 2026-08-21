from typing import List, Tuple, Optional
import face_recognition
import numpy as np
import cv2
from config import setup_logger
from src.business_logic.models import DetectionResult

logger = setup_logger(__name__)

class FaceRecognizer:
    """
    Handles real-time face detection, recognition matching, metric calculations, and HUD rendering.
    """
    def __init__(self, tolerance: float = 0.55, scale_factor: float = 0.25):
        """
        :param tolerance: Distance threshold for considering a face a match (default: 0.55)
        :param scale_factor: Downscaling factor for faster frame detection (default: 0.25)
        """
        self.tolerance = tolerance
        self.scale_factor = scale_factor

    def calculate_confidence(self, distance: float) -> float:
        """
        Convert Euclidean face distance into a normalized percentage confidence score.
        """
        if distance > self.tolerance:
            range_val = 1.0 - self.tolerance
            if range_val == 0:
                return 0.0
            linear_val = (1.0 - distance) / (range_val * 2.0)
            conf = linear_val * 100.0
        else:
            range_val = self.tolerance
            if range_val == 0:
                return 100.0
            linear_val = 1.0 - (distance / (range_val * 2.0))
            conf = linear_val * 100.0

        return float(np.clip(conf, 0.0, 100.0))

    def process_frame(
        self,
        frame: np.ndarray,
        known_encodings: List[np.ndarray],
        known_names: List[str]
    ) -> List[DetectionResult]:
        """
        Detect and identify faces in a single video frame.
        
        :param frame: BGR frame from OpenCV
        :param known_encodings: List of registered face encodings
        :param known_names: Corresponding names for encodings
        :return: List of DetectionResult objects
        """
        if frame is None or frame.size == 0:
            return []

        # Downscale for high performance
        small_frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect face locations & encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        results: List[DetectionResult] = []
        inv_scale = int(1.0 / self.scale_factor)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            is_known = False
            best_distance = 1.0
            confidence = 0.0

            if len(known_encodings) > 0:
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_idx = int(np.argmin(face_distances))
                best_distance = float(face_distances[best_match_idx])

                if best_distance <= self.tolerance:
                    name = known_names[best_match_idx]
                    is_known = True

                confidence = self.calculate_confidence(best_distance)

            # Scale coordinates back up to full frame size
            orig_top = top * inv_scale
            orig_right = right * inv_scale
            orig_bottom = bottom * inv_scale
            orig_left = left * inv_scale

            results.append(
                DetectionResult(
                    name=name,
                    confidence=round(confidence, 1),
                    distance=round(best_distance, 4),
                    location=(orig_top, orig_right, orig_bottom, orig_left),
                    is_known=is_known
                )
            )

        return results

    def draw_annotations(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        fps: Optional[float] = None,
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Draw bounding boxes, confidence badges, and live HUD metrics on the video frame.
        
        :param frame: BGR video frame to draw on
        :param detections: List of DetectionResult objects
        :param fps: Optional real-time FPS metric to display
        :param show_confidence: Whether to display confidence percentage
        :return: Annotated frame
        """
        annotated = frame.copy()

        # 1. Draw Bounding Boxes & Name Pills
        for det in detections:
            top, right, bottom, left = det.location

            # Color scheme: Emerald Green for known, Crimson Red for unknown
            box_color = (46, 204, 113) if det.is_known else (60, 60, 231)  # BGR
            text_color = (255, 255, 255)

            # Draw outer bounding box
            cv2.rectangle(annotated, (left, top), (right, bottom), box_color, 2)

            # Corner accents for a sleek futuristic HUD look
            corner_len = min(20, (right - left) // 4)
            t_thick = 3
            # Top-left
            cv2.line(annotated, (left, top), (left + corner_len, top), box_color, t_thick)
            cv2.line(annotated, (left, top), (left, top + corner_len), box_color, t_thick)
            # Top-right
            cv2.line(annotated, (right, top), (right - corner_len, top), box_color, t_thick)
            cv2.line(annotated, (right, top), (right, top + corner_len), box_color, t_thick)
            # Bottom-left
            cv2.line(annotated, (left, bottom), (left + corner_len, bottom), box_color, t_thick)
            cv2.line(annotated, (left, bottom), (left, bottom - corner_len), box_color, t_thick)
            # Bottom-right
            cv2.line(annotated, (right, bottom), (right - corner_len, bottom), box_color, t_thick)
            cv2.line(annotated, (right, bottom), (right, bottom - corner_len), box_color, t_thick)

            # Prepare label text
            if det.is_known and show_confidence:
                label_text = f"{det.name} ({det.confidence:.0f}%)"
            else:
                label_text = det.name

            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.65
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            # Position label
            if top - text_h - 12 > 0:
                bg_top = top - text_h - 12
                bg_bottom = top
                text_y = top - 6
            else:
                bg_top = bottom
                bg_bottom = bottom + text_h + 12
                text_y = bottom + text_h + 6

            cv2.rectangle(
                annotated,
                (left, bg_top),
                (left + text_w + 14, bg_bottom),
                box_color,
                cv2.FILLED
            )

            cv2.putText(
                annotated,
                label_text,
                (left + 7, text_y),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA
            )

        # 2. Draw HUD Header Overlay (FPS & Face Counts)
        if fps is not None:
            hud_text = f"FPS: {fps:.1f}  |  Faces: {len(detections)}"
            hud_font = cv2.FONT_HERSHEY_DUPLEX
            hud_scale = 0.55
            hud_thick = 1
            (hud_w, hud_h), _ = cv2.getTextSize(hud_text, hud_font, hud_scale, hud_thick)

            # Semi-transparent dark pill in top-left
            overlay = annotated.copy()
            cv2.rectangle(overlay, (10, 10), (24 + hud_w, 36), (20, 20, 20), cv2.FILLED)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

            # Green dot
            cv2.circle(annotated, (22, 23), 5, (46, 204, 113), cv2.FILLED)

            # Text
            cv2.putText(
                annotated,
                hud_text,
                (34, 28),
                hud_font,
                hud_scale,
                (240, 240, 240),
                hud_thick,
                cv2.LINE_AA
            )

        return annotated
