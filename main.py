from src.gui.main_window import FaceRecognitionApp
import logging
import sys
import flet as ft

logger = logging.getLogger(__name__)


def main(page: ft.Page):
    try:
        FaceRecognitionApp(page)
    except Exception as e:
        logger.error(f"Error occured in FaceRecognitionApp processing, check this issue: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import flet as ft
    ft.app(target=main)