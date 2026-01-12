import os
import sys

from PyQt5.QtWidgets import QApplication

# Anomalib Imports
# Set environment variable as in the original script
os.environ['TRUST_REMOTE_CODE'] = '1'

try:
    from anomalib.deploy import OpenVINOInferencer, TorchInferencer
except ImportError:
    print("Error: Anomalib not found. Please ensure it is installed in your environment.")
    sys.exit(1)

from gui.main_window import InferenceGUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InferenceGUI()
    window.show()
    sys.exit(app.exec_())
