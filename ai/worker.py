from pathlib import Path

from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QImage

from ai.inference import generate_visuals, get_inferencer, run_inference_core


class AIWorker(QObject):
    model_loaded = pyqtSignal()
    inference_finished = pyqtSignal(QImage, QImage, float, float, str)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.inferencer = None
        self.cached_image = None
        self.cached_predictions = None
        self.cached_time = None

    def load_model(self, model_path):
        try:
            self.inferencer = get_inferencer(Path(model_path))
            self.model_loaded.emit()
        except Exception as e:
            self.error_occurred.emit(f"Load Error: {str(e)}")

    def run_inference(self, image_path, thickness=3):
        if not self.inferencer:
            self.error_occurred.emit("Model not loaded.")
            return

        try:
            # Load and cache image
            self.cached_image = Image.open(image_path).convert("RGB")
            
            # Run inference and cache
            self.cached_predictions, self.cached_time = run_inference_core(
                self.cached_image, self.inferencer)
            
            self.update_contours(thickness)

        except Exception as e:
            self.error_occurred.emit(f"Inference Error: {str(e)}")

    def update_contours(self, thickness):
        if self.cached_image is None or self.cached_predictions is None:
            return

        try:
            # Generate visuals from cache
            heat_map, segmentation, inf_time, score, label = generate_visuals(
                self.cached_image, self.cached_predictions, self.cached_time, thickness
            )
            
            # Convert PIL images to QImage inside the thread for efficiency
            # We use ImageQt but ensure we copy to decouple from PIL
            qt_heat_map = ImageQt(heat_map).copy()
            qt_segmentation = ImageQt(segmentation).copy()

            self.inference_finished.emit(qt_heat_map, qt_segmentation, inf_time, score, label)
        except Exception as e:
            self.error_occurred.emit(f"Visualization Error: {str(e)}")
