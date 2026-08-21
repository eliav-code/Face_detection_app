from typing import Callable, Optional
import flet as ft
import time
from src.business_logic.face_manager import FaceManager
from src.business_logic.models import FaceProfile

class FaceGalleryView:
    """
    Visual Face Gallery and Profile Management component.
    """
    def __init__(
        self,
        page: ft.Page,
        face_manager: FaceManager,
        on_profiles_updated: Callable[[], None],
        on_request_camera_enroll: Callable[[], None],
        on_request_file_enroll: Callable[[], None]
    ):
        self.page = page
        self.face_manager = face_manager
        self.on_profiles_updated = on_profiles_updated
        self.on_request_camera_enroll = on_request_camera_enroll
        self.on_request_file_enroll = on_request_file_enroll

        self.search_query = ""
        self.profile_to_delete: Optional[FaceProfile] = None
        self.profile_to_rename: Optional[FaceProfile] = None

        self._init_dialogs()
        self._init_controls()

    def _init_dialogs(self):
        """Initialize Rename and Delete confirmation dialogs."""
        # Rename Dialog
        self.rename_input = ft.TextField(label="New Name", autofocus=True, width=280)
        self.rename_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Rename Face Profile"),
            content=ft.Column([
                ft.Text("Enter a new name for this person:"),
                self.rename_input
            ], tight=True),
            actions=[
                ft.TextButton("Cancel", on_click=self._close_rename_dialog),
                ft.ElevatedButton("Save", icon=ft.icons.SAVE, on_click=self._confirm_rename)
            ]
        )

        # Delete Confirmation Dialog
        self.delete_confirm_text = ft.Text("")
        self.delete_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Delete Face Profile"),
            content=self.delete_confirm_text,
            actions=[
                ft.TextButton("Cancel", on_click=self._close_delete_dialog),
                ft.ElevatedButton(
                    "Delete",
                    icon=ft.icons.DELETE_FOREVER,
                    bgcolor=ft.colors.ERROR,
                    color=ft.colors.WHITE,
                    on_click=self._confirm_delete
                )
            ]
        )

    def _init_controls(self):
        """Initialize UI layout components."""
        self.search_field = ft.TextField(
            prefix_icon=ft.icons.SEARCH,
            hint_text="Search registered faces by name...",
            width=280,
            on_change=self._on_search_change,
            dense=True
        )

        self.cards_grid = ft.GridView(
            expand=False,
            runs_count=3,
            max_extent=260,
            child_aspect_ratio=0.80,
            spacing=16,
            run_spacing=16,
        )

        self.camera_enroll_btn = ft.ElevatedButton(
            "Capture Webcam",
            icon=ft.icons.CAMERA_ALT,
            on_click=lambda _: self.on_request_camera_enroll()
        )

        self.file_enroll_btn = ft.ElevatedButton(
            "Upload Photo",
            icon=ft.icons.UPLOAD_FILE,
            on_click=lambda _: self.on_request_file_enroll()
        )

        self.main_container = ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        self.search_field,
                        ft.Row([
                            self.camera_enroll_btn,
                            self.file_enroll_btn
                        ], spacing=8)
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    wrap=True
                ),
                ft.Divider(),
                self.cards_grid
            ],
            spacing=14,
            scroll="AUTO",
            expand=True
        )

    def get_view(self) -> ft.Control:
        """Return the main container and refresh cards."""
        self.refresh()
        return self.main_container

    def refresh(self):
        """Re-render the face cards based on the current search query."""
        profiles = self.face_manager.search_profiles(self.search_query)
        self.cards_grid.controls.clear()

        if not profiles:
            empty_msg = (
                "No registered faces found matching your search."
                if self.search_query
                else "No faces enrolled yet. Click 'Capture Webcam' or 'Upload Photo' to add people!"
            )
            empty_state = ft.Container(
                content=ft.Column([
                    ft.Icon(ft.icons.PEOPLE_OUTLINE, size=64, color=ft.colors.OUTLINE),
                    ft.Text(empty_msg, size=15, color=ft.colors.OUTLINE, text_align="center")
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                alignment=ft.alignment.center,
                padding=40
            )
            self.cards_grid.controls.append(empty_state)
        else:
            for profile in profiles:
                self.cards_grid.controls.append(self._build_face_card(profile))

        self.page.update()

    def _build_face_card(self, profile: FaceProfile) -> ft.Control:
        """Create a styled card for a face profile."""
        # Face Avatar / Thumbnail
        if profile.thumbnail:
            avatar = ft.Container(
                content=ft.Image(
                    src_base64=profile.thumbnail,
                    width=96,
                    height=96,
                    fit="cover",
                    border_radius=48
                ),
                border=ft.border.all(2, ft.colors.PRIMARY),
                shape=ft.BoxShape.CIRCLE
            )
        else:
            initial = profile.name[0].upper() if profile.name else "?"
            avatar = ft.CircleAvatar(
                content=ft.Text(initial, size=32, weight=ft.FontWeight.BOLD, color=ft.colors.ON_PRIMARY_CONTAINER),
                radius=48,
                bgcolor=ft.colors.PRIMARY_CONTAINER
            )

        # Date formatting
        date_str = time.strftime("%b %d, %Y", time.localtime(profile.created_at))

        card = ft.Card(
            elevation=3,
            shape=ft.RoundedRectangleBorder(radius=12),
            content=ft.Container(
                padding=14,
                content=ft.Column([
                    ft.Row([avatar], alignment=ft.MainAxisAlignment.CENTER),
                    ft.Text(
                        profile.name,
                        size=16,
                        weight=ft.FontWeight.BOLD,
                        text_align="center",
                        overflow=ft.TextOverflow.ELLIPSIS,
                        max_lines=1
                    ),
                    ft.Row([
                        ft.Container(
                            content=ft.Text(f"{len(profile.encodings)} sample(s)", size=11, weight="bold"),
                            bgcolor=ft.colors.SECONDARY_CONTAINER,
                            border_radius=6,
                            padding=ft.padding.symmetric(horizontal=6, vertical=2)
                        ),
                        ft.Text(date_str, size=11, color=ft.colors.OUTLINE)
                    ], alignment=ft.MainAxisAlignment.CENTER, spacing=6),
                    ft.Divider(height=10),
                    ft.Row([
                        ft.IconButton(
                            icon=ft.icons.EDIT_OUTLINED,
                            icon_size=18,
                            tooltip="Rename",
                            on_click=lambda _, p=profile: self._open_rename_dialog(p)
                        ),
                        ft.IconButton(
                            icon=ft.icons.DELETE_OUTLINE,
                            icon_size=18,
                            icon_color=ft.colors.ERROR,
                            tooltip="Delete Profile",
                            on_click=lambda _, p=profile: self._open_delete_dialog(p)
                        )
                    ], alignment=ft.MainAxisAlignment.CENTER, spacing=8)
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=6)
            )
        )
        return card

    def _on_search_change(self, e):
        self.search_query = self.search_field.value or ""
        self.refresh()

    def _open_rename_dialog(self, profile: FaceProfile):
        self.profile_to_rename = profile
        self.rename_input.value = profile.name
        self.page.open(self.rename_dialog)
        self.page.update()

    def _close_rename_dialog(self, e):
        self.page.close(self.rename_dialog)
        self.profile_to_rename = None
        self.page.update()

    def _confirm_rename(self, e):
        if not self.profile_to_rename:
            return
        new_name = self.rename_input.value.strip()
        if not new_name:
            return

        success, msg = self.face_manager.rename_profile(self.profile_to_rename.id, new_name)
        self._close_rename_dialog(e)

        if success:
            self.refresh()
            self.on_profiles_updated()

    def _open_delete_dialog(self, profile: FaceProfile):
        self.profile_to_delete = profile
        self.delete_confirm_text.value = f"Are you sure you want to delete '{profile.name}' from the recognition database?"
        self.page.open(self.delete_dialog)
        self.page.update()

    def _close_delete_dialog(self, e):
        self.page.close(self.delete_dialog)
        self.profile_to_delete = None
        self.page.update()

    def _confirm_delete(self, e):
        if not self.profile_to_delete:
            return

        success, msg = self.face_manager.delete_face_by_id(self.profile_to_delete.id)
        self._close_delete_dialog(e)

        if success:
            self.refresh()
            self.on_profiles_updated()
