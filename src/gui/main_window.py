from typing import Optional, List
import flet as ft
import cv2
import base64
import threading
import time
import os
import copy
import numpy as np

from config import setup_logger
from src.business_logic.face_manager import FaceManager
from src.business_logic.face_recognizer import FaceRecognizer
from src.business_logic.activity_logger import ActivityLogger
from src.business_logic.models import DetectionResult
from src.utils.sound_player import (
    SoundService,
    KNOWN_SOUND_OPTIONS,
    UNKNOWN_SOUND_OPTIONS,
    play_sound_by_name
)
from src.gui.gallery_view import FaceGalleryView
from src.gui.activity_log_view import ActivityLogView

logger = setup_logger(__name__)

class FaceRecognitionApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "Face Recognition Studio"
        self.page.window.width = 960
        self.page.window.height = 840
        self.page.theme_mode = ft.ThemeMode.SYSTEM
        self.page.padding = 16

        # Core Services
        self.face_manager = FaceManager()
        self.face_recognizer = FaceRecognizer(tolerance=0.55)
        self.sound_service = SoundService(cooldown=3.0, volume=0.5)
        self.activity_logger = ActivityLogger(cooldown=5.0)

        # Multi-Threaded Streaming State
        self.stop_camera_flag = threading.Event()
        self.camera_running = False
        self.stream_thread: Optional[threading.Thread] = None
        self.detection_thread: Optional[threading.Thread] = None

        # Thread-safe frame and detection buffers
        self.frame_lock = threading.Lock()
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_detections: List[DetectionResult] = []

        # File Picker for photo enrollment
        self.file_picker = ft.FilePicker(on_result=self._on_file_picked)
        self.page.overlay.append(self.file_picker)

        # Components
        self.gallery_view = FaceGalleryView(
            page=self.page,
            face_manager=self.face_manager,
            on_profiles_updated=self._on_profiles_updated,
            on_request_camera_enroll=self.add_face_click,
            on_request_file_enroll=self._open_file_picker
        )

        self.activity_view = ActivityLogView(
            page=self.page,
            activity_logger=self.activity_logger
        )

        # Build UI Elements & Tabs
        self._init_ui_components()
        self.build_ui()

        self.page.window.on_event = self.on_window_event
        self.update_face_count()

    def _init_ui_components(self):
        """Initialize live camera, dialogs, dropdowns, and settings controls."""
        # Live Studio Status & Badge
        self.status_text = ft.Text("Ready. Click 'Start Camera' to begin.", size=14, weight="w500")
        self.status_badge = ft.Container(
            content=ft.Row([
                ft.Icon(ft.icons.FIBER_MANUAL_RECORD, size=12, color=ft.colors.GREY_400),
                self.status_text
            ], alignment="center", tight=True),
            bgcolor=ft.colors.SURFACE_VARIANT,
            padding=ft.padding.symmetric(horizontal=12, vertical=6),
            border_radius=16
        )

        # Active Image element for live streaming
        self.image_display = ft.Image(
            src_base64="",
            width=640,
            height=360,
            fit="CONTAIN"
        )

        # Visual camera placeholder before stream is started
        self.camera_placeholder = ft.Container(
            content=ft.Column([
                ft.Icon(ft.icons.VIDEOCAM_OUTLINED, size=64, color=ft.colors.OUTLINE),
                ft.Text("Camera is Inactive", size=17, weight=ft.FontWeight.BOLD, color=ft.colors.OUTLINE),
                ft.Text("Click 'Start Camera' below to launch high-speed 30+ FPS face detection", size=13, color=ft.colors.OUTLINE)
            ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=6),
            alignment=ft.alignment.center,
            width=640,
            height=360
        )

        # Video viewport box
        self.video_container = ft.Container(
            content=self.camera_placeholder,
            alignment=ft.alignment.center,
            bgcolor=ft.colors.BLACK12,
            border=ft.border.all(1, ft.colors.OUTLINE_VARIANT),
            border_radius=12,
            width=652,
            height=372
        )

        self.start_button = ft.ElevatedButton(
            "Start Camera",
            icon=ft.icons.VIDEOCAM,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                padding=14
            ),
            on_click=self.start_camera_click
        )

        self.snapshot_button = ft.ElevatedButton(
            "Take Snapshot",
            icon=ft.icons.CAMERA,
            on_click=self.take_snapshot_click
        )

        self.enroll_webcam_button = ft.ElevatedButton(
            "Enroll (Webcam)",
            icon=ft.icons.PERSON_ADD,
            on_click=self.add_face_click
        )

        self.enroll_file_button = ft.ElevatedButton(
            "Upload Photo",
            icon=ft.icons.UPLOAD_FILE,
            on_click=lambda _: self._open_file_picker()
        )

        self.delete_dropdown_button = ft.ElevatedButton(
            "Delete Face",
            icon=ft.icons.PERSON_REMOVE,
            color=ft.colors.ERROR,
            on_click=self.open_delete_dropdown_dialog
        )

        self.sound_toggle_button = ft.IconButton(
            icon=ft.icons.VOLUME_UP,
            tooltip="Toggle Alert Sounds",
            on_click=self._toggle_sound
        )

        self.face_count_badge = ft.Container(
            content=ft.Row([
                ft.Icon(ft.icons.PEOPLE, size=16),
                ft.Text(f"Known faces: {self.face_manager.get_face_count()}", size=13, weight="bold")
            ], tight=True, alignment="center"),
            bgcolor=ft.colors.PRIMARY_CONTAINER,
            padding=ft.padding.symmetric(horizontal=12, vertical=6),
            border_radius=16
        )

        # Add Face Dialog (Webcam Enrollment)
        self.name_input_to_add = ft.TextField(label="Person's Name (optional)", width=260, autofocus=True)
        self.add_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Enroll New Face via Camera"),
            content=ft.Column([
                ft.Text("Enter a name, look directly at the webcam, and click Capture:"),
                self.name_input_to_add
            ], tight=True),
            actions=[
                ft.TextButton("Cancel", on_click=self.close_add_dialog),
                ft.ElevatedButton("Capture & Enroll", icon=ft.icons.CAMERA, on_click=self.submit_adding)
            ]
        )

        # File Upload Name Dialog
        self.pending_file_path: Optional[str] = None
        self.file_name_input = ft.TextField(label="Person's Name (optional)", width=260, autofocus=True)
        self.file_add_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Enroll Face from Photo"),
            content=ft.Column([
                ft.Text("Provide an optional name for the person in this photo:"),
                self.file_name_input
            ], tight=True),
            actions=[
                ft.TextButton("Cancel", on_click=self._close_file_add_dialog),
                ft.ElevatedButton("Save Profile", icon=ft.icons.CHECK, on_click=self._submit_file_adding)
            ]
        )

        # Delete Face with Dropdown Dialog
        self.delete_dropdown = ft.Dropdown(label="Select Face to Delete", width=280)
        self.delete_dropdown_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Delete Known Face"),
            content=ft.Column([
                ft.Text("Choose a registered person to delete from the database:"),
                self.delete_dropdown
            ], tight=True),
            actions=[
                ft.TextButton("Cancel", on_click=self.close_delete_dropdown_dialog),
                ft.ElevatedButton(
                    "Delete Selected",
                    icon=ft.icons.DELETE_FOREVER,
                    bgcolor=ft.colors.ERROR,
                    color=ft.colors.WHITE,
                    on_click=self.submit_dropdown_deleting
                )
            ]
        )

        # Settings Controls - Sliders
        self.tolerance_slider = ft.Slider(
            min=0.3,
            max=0.7,
            divisions=8,
            value=self.face_recognizer.tolerance,
            label="{value}",
            on_change=self._on_tolerance_change
        )
        self.tolerance_val_text = ft.Text(f"{self.face_recognizer.tolerance:.2f}", weight="bold")

        # Sound Selection Dropdowns
        self.known_sound_dropdown = ft.Dropdown(
            label="Known Face Alert Sound",
            value=self.sound_service.known_sound,
            options=[ft.dropdown.Option(s) for s in KNOWN_SOUND_OPTIONS],
            width=320,
            on_change=self._on_known_sound_change
        )

        self.unknown_sound_dropdown = ft.Dropdown(
            label="Unknown Visitor Alert Sound",
            value=self.sound_service.unknown_sound,
            options=[ft.dropdown.Option(s) for s in UNKNOWN_SOUND_OPTIONS],
            width=320,
            on_change=self._on_unknown_sound_change
        )

    def build_ui(self):
        """Construct the modern tabbed layout."""
        # Tab 1: Live Studio View
        studio_tab = ft.Tab(
            text="Live Camera Studio",
            icon=ft.icons.VIDEOCAM,
            content=ft.Container(
                padding=16,
                content=ft.Column([
                    ft.Row([
                        self.status_badge,
                        self.face_count_badge
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    self.video_container,
                    ft.Row([
                        self.start_button,
                        self.snapshot_button,
                        self.enroll_webcam_button,
                        self.enroll_file_button,
                        self.delete_dropdown_button,
                        self.sound_toggle_button
                    ], alignment=ft.MainAxisAlignment.CENTER, spacing=8, wrap=True)
                ], alignment="start", horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=14, scroll="AUTO")
            )
        )

        # Tab 2: Visual Face Gallery
        gallery_tab = ft.Tab(
            text="Face Gallery",
            icon=ft.icons.PEOPLE_ALT,
            content=ft.Container(
                padding=16,
                content=self.gallery_view.get_view()
            )
        )

        # Tab 3: Activity & Attendance Log
        activity_tab = ft.Tab(
            text="Activity Log",
            icon=ft.icons.HISTORY,
            content=ft.Container(
                padding=16,
                content=self.activity_view.get_view()
            )
        )

        # Tab 4: Settings & Info
        settings_tab = ft.Tab(
            text="Settings & Guide",
            icon=ft.icons.SETTINGS,
            content=ft.Container(
                padding=20,
                content=ft.Column([
                    ft.Text("Recognition Configuration", size=18, weight="bold"),
                    ft.Card(
                        content=ft.Container(
                            padding=16,
                            content=ft.Column([
                                ft.Row([
                                    ft.Text("Matching Tolerance (Sensitivity):", size=14, weight="w500"),
                                    self.tolerance_val_text
                                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                                self.tolerance_slider,
                                ft.Text(
                                    "Lower values = Stricter matching (fewer false positives). Higher values = More lenient matching.",
                                    size=12,
                                    color=ft.colors.OUTLINE
                                )
                            ], spacing=6)
                        )
                    ),
                    ft.Text("Audio Alerts & Sound Customization", size=18, weight="bold"),
                    ft.Card(
                        content=ft.Container(
                            padding=16,
                            content=ft.Column([
                                ft.Text("Choose custom audio tones for detection events:", size=13, color=ft.colors.OUTLINE),
                                ft.Row([
                                    self.known_sound_dropdown,
                                    ft.IconButton(
                                        icon=ft.icons.PLAY_CIRCLE_FILL,
                                        icon_size=32,
                                        icon_color=ft.colors.PRIMARY,
                                        tooltip="Preview Known Face Sound",
                                        on_click=lambda _: self.sound_service.play_known_preview()
                                    )
                                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                                ft.Row([
                                    self.unknown_sound_dropdown,
                                    ft.IconButton(
                                        icon=ft.icons.PLAY_CIRCLE_FILL,
                                        icon_size=32,
                                        icon_color=ft.colors.ERROR,
                                        tooltip="Preview Unknown Visitor Sound",
                                        on_click=lambda _: self.sound_service.play_unknown_preview()
                                    )
                                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
                            ], spacing=12)
                        )
                    ),
                    ft.Text("Application Guide", size=18, weight="bold"),
                    ft.Card(
                        content=ft.Container(
                            padding=16,
                            content=ft.Column([
                                ft.Text("• 30+ FPS Live HUD: Real-time recognition, match percentage, and live FPS counter.", size=13),
                                ft.Text("• Sound Customization: Pick different sound effects from dropdowns and preview them live.", size=13),
                                ft.Text("• Take Snapshot: Save high-resolution photos of the camera stream to 'snapshots/'.", size=13),
                                ft.Text("• Activity Log: View real-time visitor attendance and export records to CSV.", size=13),
                                ft.Text("• Auto-Alert Snapshots: Unknown faces are automatically photographed and stored in 'alerts/'.", size=13),
                                ft.Text("• Face Gallery: Manage registered people, rename profiles, or delete with confirmation.", size=13)
                            ], spacing=6)
                        )
                    )
                ], spacing=16, scroll="AUTO")
            )
        )

        self.tabs = ft.Tabs(
            selected_index=0,
            animation_duration=250,
            tabs=[studio_tab, gallery_tab, activity_tab, settings_tab],
            expand=True
        )

        self.page.add(
            ft.Column([
                ft.Row([
                    ft.Text(
                        value="Face Recognition Studio",
                        size=26,
                        weight=ft.FontWeight.BOLD,
                        selectable=True
                    )
                ], alignment="center"),
                ft.Container(self.tabs, expand=True)
            ], expand=True, spacing=10)
        )

    def _toggle_sound(self, e):
        """Toggle alert sound playback on/off."""
        is_enabled = self.sound_service.toggle_sound()
        self.sound_toggle_button.icon = ft.icons.VOLUME_UP if is_enabled else ft.icons.VOLUME_OFF
        self.sound_toggle_button.tooltip = "Sound Alerts Enabled" if is_enabled else "Sound Alerts Muted"
        self.page.update()

    def _on_tolerance_change(self, e):
        val = round(self.tolerance_slider.value, 2)
        self.face_recognizer.tolerance = val
        self.tolerance_val_text.value = f"{val:.2f}"
        self.page.update()

    def _on_known_sound_change(self, e):
        if self.known_sound_dropdown.value:
            self.sound_service.set_known_sound(self.known_sound_dropdown.value)
            self.sound_service.play_known_preview()

    def _on_unknown_sound_change(self, e):
        if self.unknown_sound_dropdown.value:
            self.sound_service.set_unknown_sound(self.unknown_sound_dropdown.value)
            self.sound_service.play_unknown_preview()

    def update_status_text(self, message: str, is_active: bool = False):
        """Update live status badge."""
        self.status_text.value = message
        self.status_badge.content.controls[0].color = ft.colors.GREEN if is_active else ft.colors.GREY_400
        self.page.update()

    def update_face_count(self):
        """Refresh known faces count display in badge."""
        self.face_count_badge.content.controls[1].value = f"Known faces: {self.face_manager.get_face_count()}"
        self.page.update()

    def _on_profiles_updated(self):
        """Callback when gallery modifies profiles (rename/delete)."""
        self.update_face_count()

    def _open_file_picker(self):
        """Open system file picker for selecting face photo."""
        self.file_picker.pick_files(
            dialog_title="Select a Face Photo",
            allowed_extensions=["jpg", "jpeg", "png", "bmp", "webp"]
        )

    def _on_file_picked(self, e: ft.FilePickerResultEvent):
        """Handle picked photo file."""
        if not e.files or len(e.files) == 0:
            return

        file_path = e.files[0].path
        if not file_path:
            return

        self.pending_file_path = file_path
        self.file_name_input.value = ""
        self.page.open(self.file_add_dialog)
        self.page.update()

    def _close_file_add_dialog(self, e):
        self.page.close(self.file_add_dialog)
        self.pending_file_path = None
        self.page.update()

    def _submit_file_adding(self, e):
        """Enroll face from selected image file."""
        self.page.close(self.file_add_dialog)
        self.page.update()

        if not self.pending_file_path:
            return

        name = self.file_name_input.value.strip() or None
        file_path = self.pending_file_path
        self.pending_file_path = None

        self.update_status_text("Processing image file...")

        def _do_file_add():
            success, msg, _ = self.face_manager.add_face_from_image_file(file_path=file_path, name=name)
            self.update_status_text(msg)
            if success:
                self.gallery_view.refresh()
                self.update_face_count()

        threading.Thread(target=_do_file_add, daemon=True).start()

    # -------------------------------------------------------------
    # 🚀 HIGH-SPEED MULTI-THREADED VIDEO PIPELINE (30+ FPS)
    # -------------------------------------------------------------
    def _recognition_worker_loop(self) -> None:
        """
        Background worker thread: runs face detection & recognition in parallel
        without blocking the camera capture frame rate.
        """
        while not self.stop_camera_flag.is_set():
            frame_to_process = None
            with self.frame_lock:
                if self.latest_raw_frame is not None:
                    frame_to_process = self.latest_raw_frame.copy()

            if frame_to_process is not None:
                try:
                    known_encodings = self.face_manager.get_known_encodings()
                    known_names = self.face_manager.get_known_names()
                    detections = self.face_recognizer.process_frame(frame_to_process, known_encodings, known_names)

                    # Atomically update detections
                    self.latest_detections = detections

                    # Process logging & sound alerts for detected faces
                    for det in detections:
                        # Log to activity logger (auto-snapshots unknown visitor)
                        new_entry = self.activity_logger.log_detection(det, frame_to_process)
                        if new_entry:
                            # Play sound alert asynchronously
                            self.sound_service.queue_alert(
                                "known" if det.is_known else "unknown",
                                person_name=det.name
                            )

                except Exception as ex:
                    logger.error(f"Error in async recognition worker: {ex}")

            # Run detection roughly every ~100ms to balance CPU & responsiveness
            time.sleep(0.1)

    def _camera_capture_and_render_loop(self) -> None:
        """
        Main video thread: reads webcam at native 30+ FPS, renders latest HUD annotations,
        and streams to GUI smoothly.
        """
        self.sound_service.start()

        # Open camera using DirectShow on Windows for instant initialization
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.error("Unable to access webcam device.")
                self.update_status_text("Error: Unable to access camera.")
                self.camera_running = False
                self.start_button.text = "Start Camera"
                self.start_button.icon = ft.icons.VIDEOCAM
                self.video_container.content = self.camera_placeholder
                self.page.update()
                return

        # Start async recognition worker thread
        self.detection_thread = threading.Thread(target=self._recognition_worker_loop, daemon=True)
        self.detection_thread.start()

        self.video_container.content = self.image_display
        self.update_status_text("Live — 30+ FPS Face Recognition Active", is_active=True)
        self.page.update()
        logger.info("Webcam pipeline started at full speed.")

        # FPS calculation variables
        prev_time = time.time()
        fps_smooth = 30.0

        while not self.stop_camera_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from webcam.")
                self.update_status_text("Error: Failed to read camera frame.")
                break

            # Calculate instantaneous rolling FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                inst_fps = 1.0 / dt
                fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps

            # Share latest frame with background recognition worker
            with self.frame_lock:
                self.latest_raw_frame = frame

            try:
                # Copy latest detections atomically
                detections = copy.copy(self.latest_detections)

                # Draw HUD annotations & FPS overlay
                annotated_frame = self.face_recognizer.draw_annotations(
                    frame,
                    detections,
                    fps=fps_smooth,
                    show_confidence=True
                )

                # Resize for display
                display_frame = cv2.resize(annotated_frame, (640, 360))
                success, buffer = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if success:
                    img_b64 = base64.b64encode(buffer).decode("utf-8")
                    self.image_display.src_base64 = img_b64

                    # Safe update: only update if image is mounted on the page
                    if self.image_display.page is not None:
                        self.image_display.update()

            except Exception as ex:
                logger.error(f"Error in video render loop: {ex}")

            time.sleep(0.015)  # Frame pacing for smooth ~35-40 FPS

        cap.release()
        self.sound_service.stop()
        self.image_display.src_base64 = ""
        self.video_container.content = self.camera_placeholder
        self.update_status_text("Camera stopped. Ready.", is_active=False)
        self.page.update()
        logger.info("Webcam pipeline stopped.")

    def start_camera_click(self, e):
        """Handle Start/Stop Camera toggle."""
        if not self.camera_running:
            self.stop_camera_flag.clear()
            self.camera_running = True
            self.start_button.text = "Stop Camera"
            self.start_button.icon = ft.icons.VIDEOCAM_OFF
            self.page.update()

            self.stream_thread = threading.Thread(target=self._camera_capture_and_render_loop, daemon=True)
            self.stream_thread.start()
        else:
            self.stop_camera_flag.set()
            self.camera_running = False
            self.start_button.text = "Start Camera"
            self.start_button.icon = ft.icons.VIDEOCAM
            self.video_container.content = self.camera_placeholder
            self.image_display.src_base64 = ""
            self.page.update()

    def take_snapshot_click(self, e):
        """Manually save a high-resolution snapshot from the live camera stream."""
        if not self.camera_running or self.latest_raw_frame is None:
            snack = ft.SnackBar(content=ft.Text("Please start the camera before taking a snapshot."), bgcolor=ft.colors.ERROR)
            self.page.overlay.append(snack)
            snack.open = True
            self.page.update()
            return

        frame_to_save = None
        with self.frame_lock:
            if self.latest_raw_frame is not None:
                frame_to_save = self.latest_raw_frame.copy()

        if frame_to_save is not None:
            success, path = self.activity_logger.save_manual_snapshot(frame_to_save)
            snack = ft.SnackBar(
                content=ft.Text(f"Snapshot saved: {os.path.basename(path)}"),
                bgcolor=ft.colors.GREEN if success else ft.colors.ERROR,
                action="Open Folder" if success else None,
                on_action=lambda _: os.startfile(os.path.abspath(self.activity_logger.snapshots_dir)) if os.name == 'nt' else None
            )
            self.page.overlay.append(snack)
            snack.open = True
            self.page.update()

    def add_face_click(self, e=None):
        """Open Add Face dialog for webcam capture."""
        if self.camera_running:
            self.update_status_text("Please stop the live camera before enrolling via webcam.")
            return

        self.name_input_to_add.value = ""
        self.page.open(self.add_dialog)
        self.page.update()

    def close_add_dialog(self, e):
        self.name_input_to_add.value = ""
        self.page.close(self.add_dialog)
        self.page.update()

    def submit_adding(self, e):
        """Enroll face through webcam capture."""
        self.page.close(self.add_dialog)
        self.page.update()

        name = self.name_input_to_add.value.strip() or None
        self.update_status_text("Capturing face... Please look at the camera.")

        def _do_add():
            success, message = self.face_manager.capture_and_add_face(name=name)
            self.update_status_text(message)
            if success:
                self.gallery_view.refresh()
                self.update_face_count()

        threading.Thread(target=_do_add, daemon=True).start()

    def open_delete_dropdown_dialog(self, e):
        """Open delete dialog populated with dropdown of all known face names."""
        known_names = self.face_manager.list_known_faces()
        if not known_names:
            self.update_status_text("No registered faces in database to delete.")
            return

        self.delete_dropdown.options = [ft.dropdown.Option(name) for name in known_names]
        self.delete_dropdown.value = known_names[0]
        self.page.open(self.delete_dropdown_dialog)
        self.page.update()

    def close_delete_dropdown_dialog(self, e):
        self.page.close(self.delete_dropdown_dialog)
        self.page.update()

    def submit_dropdown_deleting(self, e):
        """Delete selected person from dropdown."""
        selected_name = self.delete_dropdown.value
        self.page.close(self.delete_dropdown_dialog)
        self.page.update()

        if not selected_name:
            return

        success, message = self.face_manager.delete_face_by_name(selected_name)
        self.update_status_text(message)
        if success:
            self.gallery_view.refresh()
            self.update_face_count()

    def on_window_event(self, e: ft.WindowEvent):
        """Handle window close event and clean shutdown."""
        if e.data == "close":
            self.stop_camera_flag.set()
            self.camera_running = False
            self.sound_service.stop()
            try:
                self.face_manager.save_known_faces()
            except Exception as ex:
                logger.error(f"Error saving faces on window close: {ex}")