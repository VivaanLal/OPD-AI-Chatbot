import sys
import cv2
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QTextEdit, QSlider
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap


class OPDApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI OPD Triage Chatbot")
        self.setGeometry(200, 100, 900, 700)

        # Layouts
        main_layout = QVBoxLayout()
        cam_layout = QHBoxLayout()
        control_layout = QHBoxLayout()

        # Camera label
        self.cam_label = QLabel()
        self.cam_label.setFixedSize(640, 480)
        self.cam_label.setStyleSheet("background-color: black;")
        cam_layout.addWidget(self.cam_label)

        # Output panel
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setStyleSheet("font-size: 15px;")
        cam_layout.addWidget(self.output)

        # Pain slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 10)
        self.slider.setValue(0)
        self.slider.setFixedWidth(200)

        # Buttons
        self.scan_btn = QPushButton("Scan Injury")
        self.exit_btn = QPushButton("Exit")

        control_layout.addWidget(QLabel("Pain Level:"))
        control_layout.addWidget(self.slider)
        control_layout.addWidget(self.scan_btn)
        control_layout.addWidget(self.exit_btn)

        # Combine layouts
        main_layout.addLayout(cam_layout)
        main_layout.addLayout(control_layout)
        self.setLayout(main_layout)

        # Connect events
        self.scan_btn.clicked.connect(self.scan_injury)
        self.exit_btn.clicked.connect(self.close)

        # Camera initialization
        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

        # Timer for live feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        self.current_frame = frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.cam_label.setPixmap(QPixmap.fromImage(qimg))

    def scan_injury(self):
        pain = self.slider.value()
        self.output.append(f"\n🩺 Scan Triggered\nPain Level: {pain}/10")
        self.output.append("Analyzing image...\n")

    def closeEvent(self, event):
        if hasattr(self, 'cap'):
            self.cap.release()
        return super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OPDApp()
    window.show()
    sys.exit(app.exec())