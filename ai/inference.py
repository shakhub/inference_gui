import time
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np
from anomalib.deploy import OpenVINOInferencer, TorchInferencer
from anomalib.visualization.image.functional import (
    overlay_image,
    visualize_anomaly_map,
)
from PIL import Image


def get_inferencer(weight_path: Path) -> OpenVINOInferencer | TorchInferencer:
    """Parse args and open inferencer."""
    extension = weight_path.suffix
    inferencer: OpenVINOInferencer | TorchInferencer
    module = import_module("anomalib.deploy")
    
    if extension in {".pt", ".pth", ".ckpt"}:
        torch_inferencer = module.TorchInferencer
        # Using device='auto' as in inference.py, or standard initialization
        inferencer = torch_inferencer(path=weight_path)

    elif extension in {".onnx", ".bin", ".xml"}:
        openvino_inferencer = module.OpenVINOInferencer
        inferencer = openvino_inferencer(path=weight_path)

    else:
        raise ValueError(f"Model extension {extension} is not supported.")

    return inferencer

def run_inference_core(image: Image.Image, inferencer: OpenVINOInferencer | TorchInferencer):
    """Run pure inference without visualization."""
    start_time = time.time()
    predictions = inferencer.predict(image=image)
    inference_time = time.time() - start_time
    return predictions, inference_time

def generate_visuals(image: Image.Image, predictions, inference_time: float, thickness: int = 3):
    """Generate visualizations from existing predictions."""
    heat_map = image
    if predictions.anomaly_map is not None:
        anomaly_map = visualize_anomaly_map(predictions.anomaly_map)
        heat_map = overlay_image(base=image, overlay=anomaly_map)

    segmentation = image
    if predictions.pred_mask is not None:
        # Custom contour visualization for variable thickness
        pred_mask = predictions.pred_mask
        if hasattr(pred_mask, "cpu"):
            pred_mask = pred_mask.cpu().numpy()
        
        # Ensure contiguous array
        pred_mask = np.ascontiguousarray(pred_mask)
        
        # Scale to uint8 0-255
        if pred_mask.dtype != np.uint8:
             # Assuming mask is 0-1 or boolean, scale to 255
            pred_mask = (pred_mask * 255).astype(np.uint8)

        # Handle dimensions (H, W) or (1, H, W)
        if pred_mask.ndim == 3:
             pred_mask = pred_mask.squeeze()
        
        # Resize mask if it doesn't match image size (common in some configs)
        if pred_mask.shape[:2] != (image.height, image.width)[::-1]:
             pred_mask = cv2.resize(pred_mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST)

        # Create overlay
        image_np = np.array(image)
        # Find contours
        contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw ontours: Red color (255, 0, 0)
        # Verify image is RGB
        if image_np.ndim == 2: # Grayscale
             image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
             
        cv2.drawContours(image_np, contours, -1, (255, 0, 0), int(thickness))
        
        segmentation = Image.fromarray(image_np)

    pred_score = 0.0
    if predictions.pred_score is not None:
        score = predictions.pred_score
        if hasattr(score, "item"):
            pred_score = score.item()
        else:
            pred_score = float(score)

    pred_label = "Normal" if pred_score < 0.5 else "Anomaly"

    return heat_map, segmentation, inference_time, pred_score, pred_label

def infer_logic(image: Image.Image, inferencer: OpenVINOInferencer | TorchInferencer, thickness: int = 3):
    """Wrapper for backward compatibility or single-shot runs."""
    preds, time_taken = run_inference_core(image, inferencer)
    return generate_visuals(image, preds, time_taken, thickness)
