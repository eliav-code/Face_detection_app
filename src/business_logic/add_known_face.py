"""
Backward-compatible adapter for FaceAdder, delegating to FaceManager.
"""
from typing import List, Any, Tuple, Optional
import numpy as np
from config import setup_logger
from src.business_logic.face_manager import FaceManager

logger = setup_logger(__name__)

class FaceAdder:
    """
    Adapter class providing backward compatibility for existing code using FaceAdder.
    Delegates underlying operations to FaceManager.
    """
    def __init__(self, data_file: str = "known_faces.pkl", tolerance: float = 0.4):
        self.manager = FaceManager(data_file=data_file, tolerance=tolerance)
        self.data_file = data_file
        self.tolerance = tolerance

    def is_duplicate_face(self, new_encoding: np.ndarray, known_encodings: Optional[List[np.ndarray]] = None) -> bool:
        is_dup, _ = self.manager.is_duplicate_face(new_encoding, tolerance=self.tolerance)
        return is_dup

    def capture_face_from_camera(self) -> Tuple[bool, Any]:
        success, data, _ = self.manager.capture_face_from_camera()
        return success, data

    def add_face_to_database(
        self,
        face_encoding: np.ndarray,
        name: Optional[str],
        known_encodings: List[Any],
        known_names: List[str]
    ) -> Tuple[bool, str]:
        success, msg, _ = self.manager.add_face(name=name, face_encoding=face_encoding)
        if success:
            # Sync caller's lists
            known_encodings.clear()
            known_encodings.extend(self.manager.get_known_encodings())
            known_names.clear()
            known_names.extend(self.manager.get_known_names())
        return success, msg

    def capture_and_add_face(
        self,
        name: Optional[str],
        known_encodings: Optional[List[Any]] = None,
        known_names: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        success, msg = self.manager.capture_and_add_face(name=name)
        if success and known_encodings is not None and known_names is not None:
            known_encodings.clear()
            known_encodings.extend(self.manager.get_known_encodings())
            known_names.clear()
            known_names.extend(self.manager.get_known_names())
        return success, msg

    def save_known_faces(self, known_encodings: Optional[List] = None, known_names: Optional[List[str]] = None) -> None:
        self.manager.save_known_faces()

    def load_known_faces(self) -> Tuple[List[np.ndarray], List[str]]:
        return self.manager.load_known_faces()

    def get_face_count(self) -> int:
        return self.manager.get_face_count()

    def delete_face(
        self,
        name: str,
        known_encodings: Optional[List] = None,
        known_names: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        success, msg = self.manager.delete_face_by_name(name)
        if success and known_encodings is not None and known_names is not None:
            known_encodings.clear()
            known_encodings.extend(self.manager.get_known_encodings())
            known_names.clear()
            known_names.extend(self.manager.get_known_names())
        return success, msg

    def list_known_faces(self) -> List[str]:
        return self.manager.list_known_faces()