import face_recognition
import numpy as np
import cv2

class FaceAdder:
    def __init__(self, known_face_encodings, update_status_func):
        """
        :param known_face_encodings: List of known face encodings (shared with main app)
        :param update_status_func: Function to update status text in the UI
        """
        self.known_face_encodings = known_face_encodings
        self.update_status = update_status_func

    def is_duplicate_face(self, new_encoding, tolerance=0.6):
        if not self.known_face_encodings:
            return False
        distances = face_recognition.face_distance(self.known_face_encodings, new_encoding)
        return np.any(distances <= tolerance)

    def capture_and_add_face(self, name=None):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        print("hello")

        if not ret:
            self.update_status("Error capturing frame.")
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_frame)

        if not encodings:
            self.update_status("No face detected.")
            return

        new_encoding = encodings[0]

        if self.is_duplicate_face(new_encoding):
            self.update_status("This face is already known.")
        else:
            self.known_face_encodings.append(new_encoding)
            # Here you can save the encoding and the name to disk if needed
            self.update_status(f"New face added{' as ' + name if name else ''}.")
