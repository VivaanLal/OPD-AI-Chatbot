import io
import sys
import cv2
import numpy as np
from PIL import Image
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QTextEdit, QSlider,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox
)
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtCore import Qt, QTimer


# ---------- Simple Image Analysis ----------
def analyze_frame(frame):
    """Very simple AI-ish detection using OpenCV."""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Redness mask
    mask1 = cv2.inRange(hsv, (0, 60, 40), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (160, 60, 40), (179, 255, 255))
    red_mask = cv2.bitwise_or(mask1, mask2)
    redness = np.count_nonzero(red_mask) / red_mask.size

    # Bruises (dark patches)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bruise = any(cv2.contourArea(c) > 2000 for c in contours)

    # Swelling (area of largest contour)
    swelling = 0
    if contours:
        largest = max(contours, key=cv2.contourArea)
        swelling = cv2.contourArea(largest) / (frame.shape[0] * frame.shape[1])

    # Create preview
    overlay = cv2.addWeighted(frame, 0.7, cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
    preview = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    return redness, bruise, swelling, preview


# ---------- GUI App ----------
class OPDSimple(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple OPD AI Scanner")
        self.setMinimumSize(1000, 700)

        # Camera view
        self.cam_label = QLabel()
        self.cam_label.setFixedSize(640, 480)
        self.cam_label.setStyleSheet("background:black;")

        # Output
        self.output = QTextEdit()
        self.output.setFont(QFont("Consolas", 11))
        self.output.setReadOnly(True)

        # Pain slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 10)

        # Buttons
        self.scan_btn = QPushButton("Scan")
        self.exit_btn = QPushButton("Exit")

        # Layout
        left = QVBoxLayout()
        left.addWidget(self.cam_label)
        left.addWidget(QLabel("Pain Level:"))
        left.addWidget(self.slider)
        left.addWidget(self.scan_btn)

        right = QVBoxLayout()
        right.addWidget(self.output)
        right.addWidget(self.exit_btn)

        main = QHBoxLayout()
        main.addLayout(left)
        main.addLayout(right)

        self.setLayout(main)

        # Camera init
        self.cap = self.init_camera()
        self.current_frame = None

        # Timer for camera
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(30)

        # Connect
        self.scan_btn.clicked.connect(self.scan)
        self.exit_btn.clicked.connect(self.close)

    def init_camera(self):
        """Open camera using a safe backend."""
        for backend in [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                print("Camera opened with backend:", backend)
                return cap
        QMessageBox.critical(self, "Camera", "Could not open camera.")
        return None

    def update_camera(self):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        self.current_frame = frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
        self.cam_label.setPixmap(QPixmap.fromImage(qimg))

    def scan(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "Error", "Camera not ready yet.")
            return

        redness, bruise, swelling, preview = analyze_frame(self.current_frame)
        pain = self.slider.value()

        # Convert preview to display
        buf = io.BytesIO()
        preview.save(buf, format="PNG")
        qimg = QImage.fromData(buf.getvalue())
        self.cam_label.setPixmap(QPixmap.fromImage(qimg).scaled(640, 480, Qt.KeepAspectRatio))

        # Simple triage logic
        if bruise:
            triage = "Bruise detected. Apply ice + rest."
        elif redness > 0.15:
            triage = "Heavy redness → possible inflammation."
        elif pain >= 7 and redness < 0.02:
            triage = "High pain but no injury → X-ray recommended."
        else:
            triage = "Mild signs → RICE protocol."

        # Output text
        result = f"""
=== Scan Report ===
Pain: {pain}
Redness: {redness*100:.2f}%
Bruise: {bruise}
Swelling: {swelling:.4f}

Triage Suggestion:
{triage}
"""
        self.output.setText(result)


    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = OPDSimple()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()