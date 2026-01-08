# Anomalib Inference GUI

A modular, user-friendly Graphical User Interface (GUI) for performing anomaly detection inference using [Anomalib](https://github.com/openvinotoolkit/anomalib). This application visualizes anomaly maps and segmentation masks, allowing for real-time interaction with inference results.

## Features

- **Model Support**: Load `.pt`, `.pth`, `.ckpt` (Torch) and `.onnx`, `.bin`, `.xml` (OpenVINO) models.
- **Interactive Visualization**:
  - View input image, heat map, and segmentation mask side-by-side.
  - Toggle visibility of heat maps and diagnostics.
- **Real-time Adjustments**:
  - **Decision Threshold Slider**: Dynamically adjust the anomaly score threshold to classify results as "Normal" or "Anomaly".
  - **Contour Thickness Slider**: Adjust the thickness of the segmentation contour for better visibility on high-resolution images.
- **Performance Optimized**: 
  - Separates heavy inference computation from visualization.
  - Caches predictions for instant visualization updates when changing display parameters.
  - Uses threading to keep the UI responsive during inference.
- **Cross-Platform**: Designed to run on macOS, Linux, and Windows.

## Installation

### Prerequisites
- Python 3.8+
- [Anomalib](https://github.com/openvinotoolkit/anomalib) installed in your environment.

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd inference_gui
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure you have `anomalib` installed. Depending on your installation source, you might need to install it separately or it might be covered by your existing environment.*

## Usage

1. **Run the Application:**
   ```bash
   python3 main.py
   ```

2. **Workflow:**
   - **Load Model**: Click **Change Model** in the left sidebar and select your trained model weights file.
   - **Upload Image**: Click **Upload Image** to select an image for inference.
   - **Run Inference**: Click **Run Inference**.
   - **Analyze Results**:
     - The "Normal/Anomaly" status and score will appear.
     - Use the **Decision Threshold** slider to fine-tune the classification sensitivity.
     - Use the **Contour Thickness** slider to adjust the visualization of the defect boundary without re-running inference.

## Project Structure

```text
inference_gui/
├── main.py              # Application entry point
├── requirements.txt     # Python dependencies
├── gui/                 # User Interface Logic
│   ├── main_window.py   # Main window layout and interaction logic
│   └── widgets.py       # Custom widgets (e.g., responsive image labels)
└── ai/                  # Backend Logic
    ├── inference.py     # Core inference and visualization functions using Anomalib/OpenCV
    └── worker.py        # QThread worker for handling background tasks and caching
```

## Troubleshooting

- **"Anomalib not found"**: Ensure your Python environment has access to the `anomalib` package. 
- **Application Slowness**: The application caches inference results. Only the first run on a new image is compute-intensive. Adjusting sliders afterwards uses cached data for instant updates.
