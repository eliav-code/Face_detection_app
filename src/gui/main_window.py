from ctypes import alignment
from multiprocessing.pool import CLOSE
from re import L
import flet as ft
import cv2
import base64
import threading
import time
import face_recognition
import numpy as np
from src.business_logic.add_known_face import FaceAdder
from src.utils.sound_player import play_siren_sound, play_wonderful_sound

class FaceRecognitionApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "Real-Time Face Recognition"
        self.page.window.width = 800
        self.page.window.height = 700
        self.page.scroll = "AUTO"

        self.stop_camera_flag = threading.Event()
        self.camera_running = False

        # Face recognition data - these will be managed by business logic
        self.known_face_encodings = []
        self.known_face_names = []

        # UI elements
        self.status_text = ft.Text("Click a button to begin.", size=16, selectable=True)
        self.image_display = ft.Image(src="", width=640, height=360, fit="CONTAIN")
        self.start_button = ft.ElevatedButton("Start Camera", on_click=self.start_camera_click)

        # Delete face UI
        self.name_input_to_delete = ft.TextField(label="Enter name to delete")
        self.delete_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Delete known face", selectable=True),
            content=self.name_input_to_delete,
            actions=[
                ft.TextButton("Cancel", on_click=self.close_delete_dialog),
                ft.TextButton("Submit", on_click=self.submit_deleting)
            ]
        )
        self.delete_face_button = ft.ElevatedButton("Delete known face", on_click=self.delete_face_click)

        # Add face UI
        self.name_input_to_add = ft.TextField(
            label="Person's Name (optional)",
            width=200,
            value=""
        )

        self.add_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Add unknown face", selectable=True),
            content=self.name_input_to_add,
            actions=[
                ft.TextButton("Cancel", on_click=self.close_add_dialog),
                ft.TextButton("Submit", on_click=self.submit_adding)
            ]
        )
        self.add_face_button = ft.ElevatedButton("Add Known Face", on_click=self.add_face_click)

        # Face count display
        self.face_count_text = ft.Text(f"Known faces: {len(self.known_face_encodings)}", size=14, selectable=True)

        # Add instructions for the app
        self.instructions_dialog = ft.AlertDialog(
            title=ft.Text("Information about the app:", selectable=True),
            content=self.info_text(),
            actions=[ft.ElevatedButton("Close dialog", icon=ft.icons.CLOSE, on_click=self.close_help_dialog)],
            modal=False # The dialog enable the use of the rest of the page.
        )
        self.help_icon_button = ft.IconButton(
            icon=ft.icons.HELP_OUTLINE,
            tooltip="Press for instructions",
            on_click=self.open_help_dialog
        )

        self.build_ui()
        self.page.window.on_event = self.on_window_event

        # Initialize FaceAdder (business logic)
        self.face_adder = FaceAdder()
        self.load_known_faces()

    def build_ui(self):
        """Build the user interface"""
        self.page.add(
            ft.Column([
                ft.Row([
                    ft.Text(
                        value="Face detector app", 
                        size=50,
                        weight=ft.FontWeight.W_900,
                        selectable=True)
                ], alignment="center"),
                ft.Container(self.image_display, alignment=ft.alignment.center),
                ft.Row([
                    self.start_button, 
                    self.add_face_button,
                    self.delete_face_button
                ], alignment="center"),
                self.status_text,
                self.face_count_text,
                ft.Divider(),
                ft.Row([
                    self.help_icon_button
                ], ft.MainAxisAlignment.END)
                
            ], alignment="center")
        )

    def update_status_text(self, message):
        """Update status text and refresh UI"""
        self.status_text.value = message
        self.page.update()

    def load_known_faces(self):
        """Load known faces using business logic"""
        try:
            encodings, names = self.face_adder.load_known_faces()
            self.known_face_encodings = encodings
            self.known_face_names = names
            self.update_face_count()
        except Exception as e:
            self.update_status_text(f"Error loading faces: {str(e)}")

    def update_face_count(self):
        """Update the face count display"""
        self.face_count_text.value = f"Known faces: {len(self.known_face_encodings)}"
        self.page.update()

    def start_camera(self):
        """Start camera stream with face recognition"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.update_status_text("Error: Unable to access the camera.")
            return

        self.update_status_text("Camera is running... Detecting faces...")

        # Process every nth frame for better performance
        process_this_frame = True

        while not self.stop_camera_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                self.update_status_text("Error: Failed to read frame.")
                break

            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Only process every other frame to save time
            if process_this_frame:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    
                    if True in matches:
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                    
                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Draw rectangles and labels on the frame
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Choose color based on recognition
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for known, Red for unknown
                ## TODO - add sound for known and unknown face
                # play_wonderful_sound() if name != "Unknown" else play_siren_sound()

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Draw label
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

            # Resize frame for display
            frame = cv2.resize(frame, (640, 360))
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            img_b64 = base64.b64encode(buffer).decode("utf-8")
            self.image_display.src_base64 = img_b64
            self.page.update()
            
            time.sleep(0.1)  # Small delay to prevent overwhelming the UI

        cap.release()
        self.update_status_text("Camera stopped.")
        time.sleep(3)
        self.update_status_text("Click a button to begin.")

    def start_camera_click(self, e):
        """Handle start/stop camera button click"""
        if not self.camera_running:
            self.stop_camera_flag.clear()
            self.camera_running = True
            self.start_button.text = "Stop Camera"
            self.page.update()
            threading.Thread(target=self.start_camera, daemon=True).start()
        else:
            self.stop_camera_flag.set()
            self.camera_running = False
            self.start_button.text = "Start Camera"
            self.image_display.src_base64 = ""  # Clear image on stop
            self.page.update()

    def add_face_click(self, e):
        """Handle add face button click"""
        if self.camera_running:
            self.update_status_text("Please stop the camera before adding a new face.")
            return
        
        self.page.dialog = self.add_dialog
        self.add_dialog.open = True
        self.page.update()

    def close_add_dialog(self, e):
        # Close the dialog and return to the previous window
        self.name_input_to_add.value = ""
        self.add_dialog.open = False
        self.page.update()

    def submit_adding(self, e):
        """
        Function for submit adding new face to db
        """
        # Close the dialog
        self.add_dialog.open = False
        self.page.update()

        self.update_status_text("Capturing face... Please look at the camera.")
        
        # Get the name from input field
        name = self.name_input_to_add.value.strip() if self.name_input_to_add.value.strip() else None
        
        ## TODO - Must to write name(no optional). Add check if self.name_input.value is not empty string.

        # Use business logic to add face
        try:
            success, message = self.face_adder.capture_and_add_face(
                name=name,
                known_encodings=self.known_face_encodings, 
                known_names=self.known_face_names
            )
            
            self.update_status_text(message)

            if success:
                # Update UI elements
                self.update_face_count()

            # Clear name input
            self.name_input_to_add.value = ""
            self.page.update()

            time.sleep(3)
            self.update_status_text("Click a button to begin.")
                
        except Exception as e:
            self.update_status_text(f"Error adding face: {str(e)}")


    def delete_face_click(self, e):
        """
        Opening delete dialog for submit a name of face to delete from db
        """
        self.page.dialog = self.delete_dialog
        self.delete_dialog.open = True
        self.page.update()

    def close_delete_dialog(self, e):
        # Close the dialog and return to the previous window
        self.name_input_to_delete.value = ""
        self.delete_dialog.open = False
        self.page.update()

    def submit_deleting(self, e):
        """
        Submit deleting known face from db
        """
        # Close the dialog
        self.delete_dialog.open = False
        self.page.update()

        name_to_delete = self.name_input_to_delete.value.strip()

        if not name_to_delete:
            self.status_text.value = "You must enter a name!!!"
            self.page.update()
            return
        
        # Call delete function from bussiness_logic
        success, message = self.face_adder.delete_face(
            name=name_to_delete,
            known_encodings=self.known_face_encodings,
            known_names=self.known_face_names
        )
        self.status_text.value = message
        self.page.update()
        
        if success:
            self.update_face_count()
        
        time.sleep(3)

        # Reset name_input_to_delete value
        self.name_input_to_delete.value = ""
        self.update_status_text("Click a button to begin.")

    def on_window_event(self, e: ft.WindowEvent):
        """Handle window events"""
        if e.data == "close":
            self.stop_camera_flag.set()
            self.camera_running = False
            # Save faces before closing using business logic
            try:
                self.face_adder.save_known_faces(self.known_face_encodings, self.known_face_names)
            except Exception as ex:
                print(f"Error saving faces on close: {ex}")
    
    def open_help_dialog(self, e):
        self.instructions_dialog.open = True
        self.page.dialog = self.instructions_dialog
        self.page.update()

    def info_text(self):
        return ft.Column([
            ft.Text("Instructions:", weight="bold", selectable=True),
            ft.Text("• Start Camera: Begin face recognition", size=14, selectable=True),
            ft.Text("• Add new Face: Capture and save a new face", size=14, selectable=True),
            ft.Text("• Delete known face: Enter the name you want to delete from db", size=14, selectable=True),
            ft.Text("• Green box = Recognized, Red box = Unknown", size=14, selectable=True),
        ])
    
    def close_help_dialog(self, e):
        self.instructions_dialog.open = False
        self.page.update()