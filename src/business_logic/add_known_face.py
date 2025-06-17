from typing import List, Any, Tuple
import face_recognition
import numpy as np
import cv2
import pickle
import os
from config import setup_logger

logger = setup_logger(__name__)

class FaceAdder:
    def __init__(self, data_file="known_faces.pkl", tolerance=0.4):
        """
        Initialize FaceAdder with configuration
        
        :param data_file: Path to the file where face data is stored
        :param tolerance: Tolerance for face comparison (lower = more strict)
        """
        self.data_file = data_file
        self.tolerance = tolerance

    def is_duplicate_face(self, new_encoding, known_encodings):
        """
        Check if a face encoding already exists in the known faces
        
        :param new_encoding: Face encoding to check
        :param known_encodings: List of known face encodings
        :return: True if duplicate, False otherwise
        """
        if not known_encodings:
            return False

        distances = face_recognition.face_distance(known_encodings, new_encoding)
        logger.info(f"The norm distances between all known faces and compared face is {distances}")
        return np.any(distances <= self.tolerance) # If there is at least one face that his difference is less than tolerance the compared face is familiar.

    def capture_face_from_camera(self):
        """
        Capture a single frame from camera and extract face encoding
        
        :return: Tuple (success, face_encoding_or_error_message)
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            return False, "Unable to access camera"
        
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return False, "Error capturing frame from camera"

        # Convert BGR to RGB for face_recognition library
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face encodings in the frame
        encodings = face_recognition.face_encodings(rgb_frame)

        if not encodings:
            return False, "No face detected. Please ensure your face is clearly visible"

        # Return the first (and hopefully only) face encoding
        return True, encodings[0]

    def add_face_to_database(self, face_encoding, name, known_encodings, known_names):
        """
        Add a new face encoding and name to the database
        
        :param face_encoding: The face encoding to add
        :param name: Name associated with the face
        :param known_encodings: List of known face encodings (will be modified)
        :param known_names: List of known face names (will be modified)
        :return: True if added successfully
        """
        # Check for duplicates
        if self.is_duplicate_face(face_encoding, known_encodings):
            return False, "This face is already in the database!"

        # Generate name if not provided
        if not name:
            name = f"Person_{len(known_encodings) + 1}"

        # Add to the lists
        known_encodings.append(face_encoding)
        known_names.append(name)

        return True, f"Face added successfully as '{name}'"

    def capture_and_add_face(self, name : str, known_encodings : List[Any], known_names : List[str]):
        """
        Complete process: capture face from camera and add to database
        
        :param name: Name for the person (optional)
        :param known_encodings: List of known face encodings
        :param known_names: List of known face names
        :return: Tuple (success, message)
        """
        # Step 1: Capture face from camera
        capture_success, face_data = self.capture_face_from_camera()
        
        if not capture_success:
            return False, face_data  # face_data contains error message

        face_encoding = face_data

        # Step 2: Add face to database
        add_success, message = self.add_face_to_database(
            face_encoding, name, known_encodings, known_names
        )

        if add_success:
            # Step 3: Save to file
            try:
                self.save_known_faces(known_encodings, known_names)
                return True, message
            except Exception as e:
                # Remove the face we just added since saving failed
                known_encodings.pop()
                known_names.pop()
                return False, f"Failed to save face data: {str(e)}"
        else:
            return False, message

    def save_known_faces(self, known_encodings, known_names):
        """
        Save known faces to file
        
        :param known_encodings: List of face encodings to save
        :param known_names: List of corresponding names to save
        """
        data = {
            "encodings": known_encodings,
            "names": known_names
        }
        
        with open(self.data_file, "wb") as f:
            pickle.dump(data, f)

    def load_known_faces(self):
        """
        Load known faces from file
        
        :return: Tuple (encodings, names)
        """
        if not os.path.exists(self.data_file):
            return [], []
        
        try:
            with open(self.data_file, "rb") as f:
                data = pickle.load(f)
                encodings = data.get("encodings", [])
                names = data.get("names", [])
                return encodings, names
        except Exception as e:
            print(f"Error loading known faces: {e}")
            return [], []

    def get_face_count(self):
        """
        Get the number of faces in the database
        
        :return: Number of faces
        """
        encodings, _ = self.load_known_faces()
        return len(encodings)

    def delete_face(self, name : str, known_encodings : List, known_names : List[str]) -> Tuple[bool, str]:
        """
        Delete a face from the database by index
        
        :param index: Index of face to delete
        :param known_encodings: List of known face encodings
        :param known_names: List of known face names
        :return: Tuple (success, message)
        """
        try:
            index = known_names.index(name)
            del known_encodings[index]
            del known_names[index]
            
            try:
                self.save_known_faces(known_encodings, known_names)
                return True, f"Deleted face '{name}' successfully"
            except Exception as e:
                return False, f"Error saving after deletion: {str(e)}"
        except Exception as e:
            logger.error(f"{name} is not in knows_names list!")
            return False, f"{name} is not in knows_names list!"

    def list_known_faces(self):
        """
        Get list of all known face names
        
        :return: List of names
        """
        _, names = self.load_known_faces()
        return names