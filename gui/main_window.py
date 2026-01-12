import os
from pathlib import Path

from PIL import Image
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ai.worker import AIWorker
from gui.widgets import FluidImageLabel


class InferenceGUI(QMainWindow):
    request_model_load = pyqtSignal(str)
    request_inference = pyqtSignal(str, int)
    request_contour_update = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anomalib Inference GUI")
        self.setGeometry(100, 100, 1200, 800)
        
        self.current_image_path = None
        self.last_pred_score = None 
        
        self.init_ai_thread()
        self.init_ui()
        
        # Trigger model loading
        QT_TIMER_DELAY = 100
        QTimer.singleShot(QT_TIMER_DELAY, self.prompt_load_model)

    def init_ai_thread(self):
        # Create Thread and Worker
        self.ai_thread = QThread()
        self.ai_worker = AIWorker()
        self.ai_worker.moveToThread(self.ai_thread)
        
        # Connect Signals
        self.request_model_load.connect(self.ai_worker.load_model)
        self.request_inference.connect(self.ai_worker.run_inference)
        self.request_contour_update.connect(self.ai_worker.update_contours)
        
        self.ai_worker.model_loaded.connect(self.on_model_loaded)
        self.ai_worker.inference_finished.connect(self.on_inference_finished)
        self.ai_worker.error_occurred.connect(self.on_ai_error)
        
        # Start Thread
        self.ai_thread.start()

    def closeEvent(self, event):
        # Clean up thread on close
        self.ai_thread.quit()
        self.ai_thread.wait()
        super().closeEvent(event)

    def init_ui(self):
        # Main Widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Main Horizontal Layout (Left Sidebar + Right Content)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # --- LEFT SIDEBAR ---
        sidebar_frame = QFrame()
        sidebar_frame.setFixedWidth(200)  # Make it smaller width
        sidebar_layout = QVBoxLayout(sidebar_frame)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(15)

        # Model Info in Sidebar
        self.model_label = QLabel("Model: None")
        self.model_label.setFont(QFont("Arial", 10))
        self.model_label.setStyleSheet("color: #666;")
        self.model_label.setWordWrap(True)
        sidebar_layout.addWidget(self.model_label)
        
        load_model_btn = QPushButton("Change Model")
        load_model_btn.clicked.connect(self.prompt_load_model)
        sidebar_layout.addWidget(load_model_btn)
        
        # Separator line in sidebar
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        sidebar_layout.addWidget(line)

        # Buttons in Sidebar
        self.load_image_btn = QPushButton("Upload Image")
        self.load_image_btn.setMinimumHeight(40)
        self.load_image_btn.clicked.connect(self.load_image)
        self.load_image_btn.setEnabled(False)  # Disabled until model loads
        sidebar_layout.addWidget(self.load_image_btn)

        self.infer_btn = QPushButton("Run Inference")
        self.infer_btn.setMinimumHeight(40)
        self.infer_btn.setStyleSheet(
            "background-color: #007AFF; color: white; font-weight: bold;")
        self.infer_btn.clicked.connect(self.run_inference)
        self.infer_btn.setEnabled(False) # Disabled until image loaded
        sidebar_layout.addWidget(self.infer_btn)
        
        # Heatmap toggle
        self.heatmap_toggle = QCheckBox("Show Heat Map")
        self.heatmap_toggle.setChecked(False)
        self.heatmap_toggle.toggled.connect(self.toggle_heatmap)
        self.heatmap_toggle.setStyleSheet("margin-top: 10px; font-size: 12px;")
        sidebar_layout.addWidget(self.heatmap_toggle)

        # Diagnostics Toggle
        self.diagnostics_toggle = QCheckBox("Show Diagnostics")
        self.diagnostics_toggle.setChecked(False)
        self.diagnostics_toggle.toggled.connect(self.toggle_diagnostics)
        self.diagnostics_toggle.setStyleSheet("margin-top: 5px; font-size: 12px;")
        sidebar_layout.addWidget(self.diagnostics_toggle)
        
        # Decision Threshold Slider
        sidebar_layout.addSpacing(20)
        self.threshold_label = QLabel("Decision Threshold: 0.50")
        self.threshold_label.setFont(QFont("Arial", 10, QFont.Bold))
        sidebar_layout.addWidget(self.threshold_label)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(50) # Default 0.5
        self.threshold_slider.valueChanged.connect(self.update_decision_state)
        sidebar_layout.addWidget(self.threshold_slider)

        # Contour Thickness Slider
        sidebar_layout.addSpacing(20)
        self.contour_label = QLabel("Contour Thickness: 3")
        self.contour_label.setFont(QFont("Arial", 10, QFont.Bold))
        sidebar_layout.addWidget(self.contour_label)

        self.contour_slider = QSlider(Qt.Horizontal)
        self.contour_slider.setMinimum(1)
        self.contour_slider.setMaximum(20)
        self.contour_slider.setValue(3)
        self.contour_slider.valueChanged.connect(self.update_contour_label)
        sidebar_layout.addWidget(self.contour_slider)

        # Push buttons to the top
        sidebar_layout.addStretch()

        # Status Bar / Progress at bottom of sidebar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        sidebar_layout.addWidget(self.progress_bar)

        main_layout.addWidget(sidebar_frame)

        # --- RIGHT CONTENT ---
        right_frame = QWidget()
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(20)
        main_layout.addWidget(right_frame)

        # 1. Metrics Section (Now at TOP) with Indicator Box
        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # Metrics Group (Text)
        self.lbl_score = QLabel("Score: -")
        self.lbl_time = QLabel("Time: -")
        self.lbl_label = QLabel("Prediction: -")
        
        meta_font = QFont("Arial", 14, QFont.Bold)
        for lbl in [self.lbl_score, self.lbl_time, self.lbl_label]:
            lbl.setFont(meta_font)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setVisible(False) # Hidden by default
            
        header_layout.addStretch()
        header_layout.addWidget(self.lbl_score)
        header_layout.addSpacing(20)
        header_layout.addWidget(self.lbl_label)
        header_layout.addSpacing(20)
        header_layout.addWidget(self.lbl_time)
        header_layout.addSpacing(30)
        
        # Indicator Box (Top Right Square)
        self.status_indicator = QLabel("READY")
        self.status_indicator.setAlignment(Qt.AlignCenter)
        self.status_indicator.setFont(QFont("Arial", 20, QFont.Bold))
        self.status_indicator.setMinimumSize(200, 200)
        self.status_indicator.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.status_indicator.setStyleSheet(
            "background-color: #cccccc; border: 1px solid #999; border-radius: 5px;")
        header_layout.addWidget(self.status_indicator)
        
        right_layout.addWidget(header_container)

        # 2. Results Section (Images)
        results_scroll = QScrollArea()
        results_scroll.setWidgetResizable(True)
        results_scroll.setFrameShape(QFrame.NoFrame)
        results_widget = QWidget()
        results_layout = QHBoxLayout(results_widget)
        
        # Placeholders for images
        self.img_label_input = self.create_image_label("Input Image")
        self.img_label_heatmap = self.create_image_label("Heat Map")
        self.img_label_heatmap.setVisible(False) # Hidden by default
        self.img_label_segment = self.create_image_label("Segmentation")

        results_layout.addWidget(self.img_label_input)
        results_layout.addWidget(self.img_label_heatmap)
        results_layout.addWidget(self.img_label_segment)
        
        results_scroll.setWidget(results_widget)
        right_layout.addWidget(results_scroll)

    def create_image_label(self, title):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        title_lbl = QLabel(title)
        title_lbl.setAlignment(Qt.AlignCenter)
        title_lbl.setFont(QFont("Arial", 10, QFont.Bold))
        title_lbl.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Fixed)
        
        # Use FluidImageLabel for responsive resizing
        img_lbl = FluidImageLabel()
        img_lbl.setAlignment(
            Qt.AlignTop | Qt.AlignHCenter)
        # Style handles border/bg
        img_lbl.setMinimumSize(200, 200) # Set reasonable minimum
        img_lbl.setText("No Image")
        
        layout.addWidget(title_lbl)
        layout.addWidget(img_lbl)
        
        # Store reference to actual image label in container for easy access
        container.image_label = img_lbl 
        return container

    def toggle_heatmap(self, checked):
        self.img_label_heatmap.setVisible(checked)

    def toggle_diagnostics(self, checked):
        for lbl in [self.lbl_score, self.lbl_time, self.lbl_label]:
            lbl.setVisible(checked)

    def prompt_load_model(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Model File", 
            os.getcwd(), 
            "Model Files (*.pt *.pth *.ckpt *.onnx *.bin *.xml);;All Files (*)"
        )
        
        if file_name:
            self.model_label.setText(f"Loading: {Path(file_name).name}...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0) # Indeterminate mode
            self.load_image_btn.setEnabled(False)
            
            # Request Worker to Load
            self.request_model_load.emit(file_name)

    def on_model_loaded(self):
        # We don't get 'inferencer' back here, it lives in the thread.
        self.progress_bar.setVisible(False)
        self.model_label.setText("Model Loaded Ready")
        self.model_label.setStyleSheet("color: green; font-weight: bold;")
        self.load_image_btn.setEnabled(True)
        QMessageBox.information(self, "Success", "Model loaded successfully!")

    def on_ai_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.load_image_btn.setEnabled(True) # Re-enable just in case
        self.infer_btn.setEnabled(True)
        self.model_label.setText("Error/Idle")
        self.model_label.setStyleSheet("color: red;")
        QMessageBox.critical(self, "Error", f"An error occurred:\n{error_msg}")

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            os.getcwd(),
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        
        if file_name:
            self.current_image_path = file_name
            self.display_image(file_name, self.img_label_input.image_label)
            self.infer_btn.setEnabled(True)
            
            # Clear previous results
            self.img_label_heatmap.image_label.clear()
            self.img_label_heatmap.image_label.setText("Waiting for inference...")
            self.img_label_segment.image_label.clear()
            self.img_label_segment.image_label.setText("Waiting for inference...")
            self.reset_metrics()

    def display_image(self, source, label_widget):
        if not isinstance(label_widget, FluidImageLabel):
            return

        # source can be Path (str), PIL.Image, or QImage
        if isinstance(source, str): # Path
            pixmap = QPixmap(source)
            label_widget.set_image(pixmap)
        elif isinstance(source, Image.Image): # PIL Image
            if source.mode != "RGBA":
                source = source.convert("RGBA")
            data = source.tobytes("raw", "RGBA")
            qim = QImage(data, source.size[0], source.size[1], QImage.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qim)
            label_widget.set_image(pixmap)
        elif isinstance(source, QImage): # QImage (from thread)
            pixmap = QPixmap.fromImage(source)
            label_widget.set_image(pixmap)
        else:
            return

    def run_inference(self):
        if not self.current_image_path:
            return

        self.infer_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.lbl_label.setText("Processing...")

        # Get thickness
        thickness = self.contour_slider.value()

        # Request Inference
        self.request_inference.emit(self.current_image_path, thickness)

    def on_inference_finished(self, heat_map, segmentation, inf_time, score, label):
        self.progress_bar.setVisible(False)
        self.infer_btn.setEnabled(True)

        # Display Images (Direct QImage support in updated display_image)
        self.display_image(heat_map, self.img_label_heatmap.image_label)
        self.display_image(segmentation, self.img_label_segment.image_label)


        # Display Metrics
        self.lbl_time.setText(f"Time: {inf_time:.4f}s")
        self.lbl_score.setText(f"Score: {score:.4f}")
        
        # Store score and update decision
        self.last_pred_score = score
        self.update_decision_state()

    def update_contour_label(self):
        val = self.contour_slider.value()
        self.contour_label.setText(f"Contour Thickness: {val}")
        self.request_contour_update.emit(val)

    def update_decision_state(self):
        # Update Slider Label
        threshold_val = self.threshold_slider.value() / 100.0
        self.threshold_label.setText(f"Decision Threshold: {threshold_val:.2f}")

        if self.last_pred_score is None:
            return

        # Determine Label based on Slider Threshold
        if self.last_pred_score >= threshold_val:
            label = "Anomaly"
            display_text = "BAD"
            color = "red"
        else:
            label = "Normal"
            display_text = "GOOD"
            color = "#00FF00" # Bright Green

        self.lbl_label.setText(f"Prediction: {label}")
        self.status_indicator.setText(display_text)
        self.status_indicator.setStyleSheet(f"background-color: {color}; border: 2px solid #666; border-radius: 5px;")
        self.lbl_label.setStyleSheet("color: black")

    def reset_metrics(self):
        self.lbl_score.setText("Score: -")
        self.lbl_label.setText("Prediction: -")
        self.lbl_time.setText("Time: -")
        self.lbl_label.setStyleSheet("color: black")
