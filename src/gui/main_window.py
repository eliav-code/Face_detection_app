import flet as ft
import cv2
import base64
import threading
import time

class FaceRecognitionApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "Real-Time Face Recognition"
        self.page.window.width = 700
        self.page.window.height = 550
        self.page.scroll = "AUTO"

        self.stop_camera_flag = threading.Event()
        self.camera_running = False  # Track camera state

        # UI elements
        self.status_text = ft.Text("Click a button to begin.", size=16)
        self.image_display = ft.Image(src="", width=640, height=360, fit="CONTAIN")
        self.start_button = ft.ElevatedButton("Start Camera", on_click=self.start_camera_click)
        self.add_face_button = ft.ElevatedButton("Add Known Face", on_click=self.add_face_click)

        self.build_ui()

        self.page.window.on_event = self.on_window_event

    def build_ui(self):
        self.page.add(
            ft.Column([
                ft.Row([self.start_button, self.add_face_button], alignment="center"),
                ft.Container(self.image_display, alignment=ft.alignment.center),
                self.status_text,
            ], alignment="center")
        )

    def start_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.status_text.value = "Error: Unable to access the camera."
            self.page.update()
            return

        self.status_text.value = "Camera is running..."
        self.page.update()

        while not self.stop_camera_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                self.status_text.value = "Error: Failed to read frame."
                self.page.update()
                break

            frame = cv2.resize(frame, (300, 200))
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            img_b64 = base64.b64encode(buffer).decode("utf-8")
            self.image_display.src_base64 = img_b64
            self.page.update()
            time.sleep(0.1)

        cap.release()
        self.status_text.value = "Camera stopped."
        self.page.update()

    def start_camera_click(self, e):
        if not self.camera_running:
            self.stop_camera_flag.clear()
            self.camera_running = True
            self.start_button.text = "Close Camera"
            self.page.update()
            threading.Thread(target=self.start_camera, daemon=True).start()
        else:
            self.stop_camera_flag.set()
            self.camera_running = False
            self.start_button.text = "Start Camera"
            self.image_display.src_base64 = ""  # Clear image on stop
            self.page.update()

    def add_face_click(self, e):
        self.status_text.value = "Add known face mode (not implemented yet)."
        self.page.update()

    def on_window_event(self, e: ft.WindowEvent):
        if e.data == "close":
            self.stop_camera_flag.set()
            self.camera_running = False

def main(page: ft.Page):
    FaceRecognitionApp(page)

ft.app(target=main)
