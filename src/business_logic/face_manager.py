from typing import List, Tuple, Optional, Any
import face_recognition
import numpy as np
import cv2
import pickle
import os
import base64
import time
from config import setup_logger
from src.business_logic.models import FaceProfile

logger = setup_logger(__name__)

class FaceManager:
    """
    Manages known face profiles, persistent storage, face enrollment, and profile customization.
    """
    def __init__(self, data_file: str = "known_faces.pkl", tolerance: float = 0.4):
        self.data_file = data_file
        self.tolerance = tolerance
        self.profiles: List[FaceProfile] = []
        self.load_known_faces()

    def get_known_encodings(self) -> List[np.ndarray]:
        """Flatten and return all known face encodings."""
        all_encodings = []
        for profile in self.profiles:
            for enc in profile.encodings:
                all_encodings.append(enc)
        return all_encodings

    def get_known_names(self) -> List[str]:
        """Return names corresponding to each encoding in get_known_encodings()."""
        all_names = []
        for profile in self.profiles:
            for _ in profile.encodings:
                all_names.append(profile.name)
        return all_names

    def get_profile_by_name(self, name: str) -> Optional[FaceProfile]:
        """Find a profile by person's name."""
        for profile in self.profiles:
            if profile.name.lower() == name.lower():
                return profile
        return None

    def get_profile_by_id(self, profile_id: str) -> Optional[FaceProfile]:
        """Find a profile by unique ID."""
        for profile in self.profiles:
            if profile.id == profile_id:
                return profile
        return None

    def search_profiles(self, query: str) -> List[FaceProfile]:
        """Search profiles by name (case-insensitive substring match)."""
        if not query or not query.strip():
            return self.profiles
        q = query.strip().lower()
        return [p for p in self.profiles if q in p.name.lower()]

    def is_duplicate_face(self, new_encoding: np.ndarray, tolerance: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        """
        Check if a face encoding already exists in the registered profiles.
        
        :param new_encoding: Face encoding to check
        :param tolerance: Tolerance threshold (lower = stricter). Defaults to self.tolerance.
        :return: Tuple (is_duplicate: bool, matching_name: Optional[str])
        """
        known_encodings = self.get_known_encodings()
        known_names = self.get_known_names()

        if not known_encodings:
            return False, None

        thresh = tolerance if tolerance is not None else self.tolerance
        distances = face_recognition.face_distance(known_encodings, new_encoding)
        min_idx = int(np.argmin(distances))
        min_dist = float(distances[min_idx])

        logger.info(f"Checking duplicate: minimum distance = {min_dist:.4f} (threshold: {thresh})")

        if min_dist <= thresh:
            return True, known_names[min_idx]

        return False, None

    def _extract_thumbnail_b64(self, frame: np.ndarray, location: Tuple[int, int, int, int]) -> str:
        """Helper to crop and resize a face region into a Base64 JPEG thumbnail."""
        top, right, bottom, left = location
        h, w, _ = frame.shape
        pad_h = int((bottom - top) * 0.25)
        pad_w = int((right - left) * 0.25)
        crop_top = max(0, top - pad_h)
        crop_bottom = min(h, bottom + pad_h)
        crop_left = max(0, left - pad_w)
        crop_right = min(w, right + pad_w)

        face_crop = frame[crop_top:crop_bottom, crop_left:crop_right]
        face_crop_resized = cv2.resize(face_crop, (150, 150))
        _, buffer = cv2.imencode(".jpg", face_crop_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return base64.b64encode(buffer).decode("utf-8")

    def capture_face_from_camera(self, camera_index: int = 0) -> Tuple[bool, Any, Optional[str]]:
        """
        Capture a single frame from the camera, extract the face encoding and create a thumbnail.
        
        :param camera_index: OpenCV camera index
        :return: Tuple (success, face_encoding_or_error_message, thumbnail_b64)
        """
        # Use CAP_DSHOW on Windows for fast device initialization
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            # Fallback without CAP_DSHOW
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                return False, "Unable to access camera", None

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return False, "Error capturing frame from camera", None

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, locations)

        if not encodings:
            return False, "No face detected. Please ensure your face is clearly visible", None

        if len(encodings) > 1:
            return False, "Multiple faces detected. Please ensure only one person is in front of the camera", None

        thumbnail_b64 = self._extract_thumbnail_b64(frame, locations[0])
        return True, encodings[0], thumbnail_b64

    def add_face(self, name: Optional[str], face_encoding: np.ndarray, thumbnail: Optional[str] = None) -> Tuple[bool, str, Optional[FaceProfile]]:
        """
        Add a new face encoding to the database.
        
        :param name: Person name
        :param face_encoding: Extracted 128-d vector
        :param thumbnail: Base64 preview thumbnail string
        :return: Tuple (success, message, profile)
        """
        is_dup, matching_name = self.is_duplicate_face(face_encoding)
        if is_dup:
            return False, f"This face is already registered as '{matching_name}'!", None

        if not name or not name.strip():
            name = f"Person_{len(self.profiles) + 1}"
        else:
            name = name.strip()

        # Check if person already exists to append new encoding
        existing_profile = self.get_profile_by_name(name)
        if existing_profile:
            existing_profile.encodings.append(face_encoding)
            if thumbnail and not existing_profile.thumbnail:
                existing_profile.thumbnail = thumbnail
            profile = existing_profile
            msg = f"Added additional sample for '{name}'"
        else:
            profile = FaceProfile(
                name=name,
                encodings=[face_encoding],
                created_at=time.time(),
                thumbnail=thumbnail
            )
            self.profiles.append(profile)
            msg = f"Face registered successfully as '{name}'"

        try:
            self.save_known_faces()
            return True, msg, profile
        except Exception as e:
            logger.error(f"Failed to save profile after addition: {e}")
            if existing_profile:
                existing_profile.encodings.pop()
            else:
                self.profiles.remove(profile)
            return False, f"Failed to save face data: {str(e)}", None

    def add_face_from_image_file(self, file_path: str, name: Optional[str]) -> Tuple[bool, str, Optional[FaceProfile]]:
        """
        Enroll a new face from an image file on disk (.jpg, .png, etc.).
        """
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}", None

        try:
            frame = cv2.imread(file_path)
            if frame is None:
                return False, "Unable to load image file. Unsupported format.", None

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb_frame)
            encodings = face_recognition.face_encodings(rgb_frame, locations)

            if not encodings:
                return False, "No face detected in the selected image.", None

            if len(encodings) > 1:
                return False, f"Found {len(encodings)} faces in the image. Please select a photo containing only 1 person.", None

            thumbnail_b64 = self._extract_thumbnail_b64(frame, locations[0])

            # If name not provided, use filename without extension
            if not name or not name.strip():
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                name = base_name.replace("_", " ").title()

            return self.add_face(name=name, face_encoding=encodings[0], thumbnail=thumbnail_b64)
        except Exception as e:
            logger.error(f"Error enrolling face from image file: {e}")
            return False, f"Error processing image: {str(e)}", None

    def capture_and_add_face(self, name: Optional[str], camera_index: int = 0) -> Tuple[bool, str]:
        """
        Convenience method: capture from camera and enroll face.
        """
        success, face_data, thumbnail = self.capture_face_from_camera(camera_index)
        if not success:
            return False, str(face_data)

        added, msg, _ = self.add_face(name=name, face_encoding=face_data, thumbnail=thumbnail)
        return added, msg

    def rename_profile(self, profile_id: str, new_name: str) -> Tuple[bool, str]:
        """
        Rename an existing face profile by ID.
        """
        profile = self.get_profile_by_id(profile_id)
        if not profile:
            return False, "Profile not found."

        new_name_clean = new_name.strip()
        if not new_name_clean:
            return False, "Name cannot be empty."

        old_name = profile.name
        profile.name = new_name_clean
        try:
            self.save_known_faces()
            logger.info(f"Renamed profile ID '{profile_id}' from '{old_name}' to '{new_name_clean}'")
            return True, f"Renamed '{old_name}' to '{new_name_clean}'"
        except Exception as e:
            profile.name = old_name
            return False, f"Failed to save changes: {str(e)}"

    def delete_face_by_name(self, name: str) -> Tuple[bool, str]:
        """
        Delete a known face profile by name.
        """
        name_clean = name.strip()
        matched = [p for p in self.profiles if p.name.lower() == name_clean.lower()]
        if not matched:
            logger.warning(f"Attempted to delete '{name_clean}', but not found in database.")
            return False, f"'{name_clean}' was not found in the database."

        for p in matched:
            self.profiles.remove(p)

        try:
            self.save_known_faces()
            logger.info(f"Deleted profile '{name_clean}' successfully.")
            return True, f"Deleted '{name_clean}' successfully."
        except Exception as e:
            logger.error(f"Error saving database after deletion: {e}")
            return False, f"Error saving after deletion: {str(e)}"

    def delete_face_by_id(self, profile_id: str) -> Tuple[bool, str]:
        """
        Delete a known face profile by unique ID.
        """
        profile = self.get_profile_by_id(profile_id)
        if not profile:
            return False, f"Profile ID '{profile_id}' not found."

        name = profile.name
        self.profiles.remove(profile)
        try:
            self.save_known_faces()
            logger.info(f"Deleted profile ID '{profile_id}' ({name}) successfully.")
            return True, f"Deleted '{name}' successfully."
        except Exception as e:
            return False, f"Error saving after deletion: {str(e)}"

    def save_known_faces(self) -> None:
        """
        Save face profiles to disk with backward-compatible format.
        """
        encodings = self.get_known_encodings()
        names = self.get_known_names()

        data = {
            "profiles": self.profiles,
            "encodings": encodings,
            "names": names
        }
        with open(self.data_file, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved {len(self.profiles)} profile(s) to {self.data_file}")

    def load_known_faces(self) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load face profiles from disk, seamlessly upgrading legacy pickle formats.
        
        :return: Tuple (legacy_encodings_list, legacy_names_list)
        """
        if not os.path.exists(self.data_file):
            self.profiles = []
            return [], []

        try:
            with open(self.data_file, "rb") as f:
                data = pickle.load(f)

            if isinstance(data, dict):
                # Check for modern profiles list
                if "profiles" in data and isinstance(data["profiles"], list):
                    self.profiles = data["profiles"]
                # Legacy format upgrade
                elif "encodings" in data and "names" in data:
                    encs = data.get("encodings", [])
                    nms = data.get("names", [])
                    self.profiles = []
                    for enc, nm in zip(encs, nms):
                        self.profiles.append(FaceProfile(name=nm, encodings=[enc]))
                    logger.info(f"Upgraded legacy known_faces.pkl: {len(self.profiles)} profile(s)")
            elif isinstance(data, list):
                self.profiles = data
            else:
                self.profiles = []

            return self.get_known_encodings(), self.get_known_names()
        except Exception as e:
            logger.error(f"Error loading known faces from {self.data_file}: {e}")
            self.profiles = []
            return [], []

    def get_face_count(self) -> int:
        """Return total number of registered profiles."""
        return len(self.profiles)

    def list_known_faces(self) -> List[str]:
        """Return unique list of registered names."""
        return [p.name for p in self.profiles]
