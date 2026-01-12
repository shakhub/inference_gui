from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QSizePolicy


class FluidImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px dashed #ccc; background-color: #eee;")
        self._pixmap = None

    def set_image(self, pixmap):
        if pixmap and not pixmap.isNull():
            self._pixmap = pixmap
            self.update_display()
        else:
            self._pixmap = None
            self.clear()
            self.setText("No Image")

    def update_display(self):
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation
            )
            super().setPixmap(scaled)

    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)
