import flet as ft
import os
import subprocess
from typing import Callable, Optional
from src.business_logic.activity_logger import ActivityLogger, ActivityLogEntry

class ActivityLogView:
    """
    Real-time Visitor Activity & Attendance Log UI component.
    """
    def __init__(self, page: ft.Page, activity_logger: ActivityLogger):
        self.page = page
        self.activity_logger = activity_logger

        self._init_controls()

    def _init_controls(self):
        """Initialize table and action buttons."""
        self.export_button = ft.ElevatedButton(
            "Export to CSV",
            icon=ft.icons.DOWNLOAD,
            on_click=self._export_csv
        )

        self.open_alerts_btn = ft.ElevatedButton(
            "Alerts Folder",
            icon=ft.icons.FOLDER_SPECIAL,
            on_click=self._open_alerts_folder
        )

        self.open_snapshots_btn = ft.ElevatedButton(
            "Snapshots Folder",
            icon=ft.icons.PHOTO_LIBRARY,
            on_click=self._open_snapshots_folder
        )

        self.clear_btn = ft.IconButton(
            icon=ft.icons.DELETE_SWEEP,
            tooltip="Clear Log History",
            icon_color=ft.colors.ERROR,
            on_click=self._clear_logs
        )

        self.count_text = ft.Text("Total Events: 0", size=14, weight="w500")

        self.table_column = ft.Column(
            controls=[],
            spacing=8,
            scroll="AUTO",
            expand=True
        )

        self.main_container = ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        self.count_text,
                        ft.Row([
                            self.export_button,
                            self.open_alerts_btn,
                            self.open_snapshots_btn,
                            self.clear_btn
                        ], spacing=8, wrap=True)
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    wrap=True
                ),
                ft.Divider(),
                self.table_column
            ],
            spacing=14,
            expand=True
        )

    def get_view(self) -> ft.Control:
        """Return the component container and refresh."""
        self.refresh()
        return self.main_container

    def refresh(self):
        """Re-render the activity log table with the latest entries."""
        entries = self.activity_logger.get_entries()
        self.count_text.value = f"Total Events: {len(entries)}"
        self.table_column.controls.clear()

        if not entries:
            empty_state = ft.Container(
                content=ft.Column([
                    ft.Icon(ft.icons.HISTORY, size=56, color=ft.colors.OUTLINE),
                    ft.Text("No visitor activity recorded yet.", size=15, color=ft.colors.OUTLINE),
                    ft.Text("Detections during camera stream will automatically appear here.", size=13, color=ft.colors.OUTLINE)
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=6),
                alignment=ft.alignment.center,
                padding=40
            )
            self.table_column.controls.append(empty_state)
        else:
            # Build Data Table
            rows = []
            for entry in entries[:100]:  # Show up to 100 recent rows
                status_color = ft.colors.GREEN if entry.is_known else ft.colors.RED
                status_text = "Recognized" if entry.is_known else "Unknown Visitor"

                status_badge = ft.Container(
                    content=ft.Text(status_text, size=12, color=ft.colors.WHITE, weight="bold"),
                    bgcolor=status_color,
                    padding=ft.padding.symmetric(horizontal=8, vertical=4),
                    border_radius=6
                )

                snapshot_cell = ft.Text("—", color=ft.colors.OUTLINE)
                if entry.snapshot_path and os.path.exists(entry.snapshot_path):
                    snapshot_cell = ft.TextButton(
                        "View Alert Photo",
                        icon=ft.icons.IMAGE,
                        on_click=lambda _, p=entry.snapshot_path: self._open_file(p)
                    )

                conf_text = f"{entry.confidence:.1f}%" if entry.is_known else "—"

                rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text(entry.time_str, size=13)),
                            ft.DataCell(ft.Text(entry.name, size=13, weight="bold")),
                            ft.DataCell(status_badge),
                            ft.DataCell(ft.Text(conf_text, size=13)),
                            ft.DataCell(snapshot_cell)
                        ]
                    )
                )

            data_table = ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("Timestamp", weight="bold")),
                    ft.DataColumn(ft.Text("Person Name", weight="bold")),
                    ft.DataColumn(ft.Text("Status", weight="bold")),
                    ft.DataColumn(ft.Text("Confidence", weight="bold")),
                    ft.DataColumn(ft.Text("Alert Snapshot", weight="bold")),
                ],
                rows=rows,
                border=ft.border.all(1, ft.colors.OUTLINE_VARIANT),
                border_radius=8,
                vertical_lines=ft.border.BorderSide(1, ft.colors.OUTLINE_VARIANT),
                heading_row_color=ft.colors.SURFACE_VARIANT
            )
            self.table_column.controls.append(data_table)

        if self.page:
            self.page.update()

    def _export_csv(self, e):
        """Export logs to CSV and show notification."""
        filename = f"attendance_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        success, msg = self.activity_logger.export_to_csv(filename)

        snack = ft.SnackBar(
            content=ft.Text(msg),
            bgcolor=ft.colors.GREEN if success else ft.colors.ERROR
        )
        self.page.overlay.append(snack)
        snack.open = True
        self.page.update()

    def _open_alerts_folder(self, e):
        """Open the alerts snapshot directory in system file explorer."""
        path = os.path.abspath(self.activity_logger.alerts_dir)
        self._open_folder(path)

    def _open_snapshots_folder(self, e):
        """Open the manual snapshots directory in system file explorer."""
        path = os.path.abspath(self.activity_logger.snapshots_dir)
        self._open_folder(path)

    def _open_folder(self, path: str):
        if os.name == 'nt':
            os.startfile(path)
        else:
            subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", path])

    def _open_file(self, file_path: str):
        abs_path = os.path.abspath(file_path)
        if os.path.exists(abs_path):
            if os.name == 'nt':
                os.startfile(abs_path)

    def _clear_logs(self, e):
        self.activity_logger.clear_entries()
        self.refresh()
