
import sys, io, cv2, numpy as np, traceback, base64, threading
from PIL import Image


OPENAI_API_KEY = "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"

# Using the new OpenAI client library
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    OPENAI_OK = True
except:
    OPENAI_OK = False


from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QTextEdit, QSlider,
    QVBoxLayout, QHBoxLayout, QMessageBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont


# ----------------------------------------------------
# SIMPLE LOCAL CV ANALYSIS
# ----------------------------------------------------
def analyze_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Redness
    m1 = cv2.inRange(hsv, (0, 60, 40), (10, 255, 255))
    m2 = cv2.inRange(hsv, (160, 60, 40), (179, 255, 255))
    red_mask = cv2.bitwise_or(m1, m2)
    redness = np.count_nonzero(red_mask) / red_mask.size

    # Bruises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thr = cv2.threshold(blur, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    bruise = any(cv2.contourArea(c) > 2200 for c in cnts)

    # Swelling
    swelling = 0
    if cnts:
        biggest = max(cnts, key=cv2.contourArea)
        swelling = cv2.contourArea(biggest) / (frame.shape[0] * frame.shape[1])

    # Preview overlay
    red_overlay = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(frame, 0.7, red_overlay, 0.3, 0)
    preview = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    return redness, bruise, swelling, preview



# ----------------------------------------------------
# OPENAI TRIAGE (CORRECTED + IMAGE SUPPORT)
# ----------------------------------------------------
def gpt_triage(image_pil, text):
    if not OPENAI_OK:
        return "OpenAI SDK not installed."

    if OPENAI_API_KEY == "" or OPENAI_API_KEY.startswith("PASTE"):
        return "Add your OpenAI API Key in the code."

    try:
        # Convert PIL → base64 data URL
        buf = io.BytesIO()
        image_pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = "data:image/png;base64," + b64

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Provide ONLY simple first-aid triage. No diagnosis."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "input_image", "image_url": data_url}
                    ]
                }
            ],
            max_tokens=200
        )

        return response.choices[0].message["content"]

    except Exception as e:
        traceback.print_exc()
        return f"[GPT ERROR] {e}"



# ----------------------------------------------------
# MAIN ONE-BUTTON UI (FREEZE-PROOF)
# ----------------------------------------------------
class OPDOne(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OPD AI Scanner — One Button")
        self.setMinimumSize(900, 680)
        self.setStyleSheet("background:#f5f5f5;")

        # CAMERA WIDGET
        self.cam = QLabel()
        self.cam.setFixedSize(640, 480)
        self.cam.setStyleSheet("background:black; border:2px solid #444; border-radius:10px;")

        # OUTPUT BOX
        self.out = QTextEdit()
        self.out.setReadOnly(True)
        self.out.setFont(QFont("Arial", 12))
        self.out.setFixedHeight(150)
        self.out.setStyleSheet("border:1px solid #aaa; padding:8px; border-radius:6px;")

        # PAIN SLIDER
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 10)
        self.slider.setStyleSheet("padding:6px;")

        # ONE BUTTON
        self.scan_btn = QPushButton("📸 Scan & AI Triage")
        self.scan_btn.setStyleSheet("""
            QPushButton {
                background:#0275d8; 
                color:white; 
                font-size:17px; 
                padding:12px; 
                border-radius:8px;
            }
            QPushButton:hover { background:#025aa5; }
        """)

        # Layout
        right = QVBoxLayout()
        right.addWidget(QLabel("Pain Level (0–10):"))
        right.addWidget(self.slider)
        right.addSpacing(20)
        right.addWidget(self.scan_btn)
        right.addStretch()

        top = QHBoxLayout()
        top.addWidget(self.cam)
        top.addLayout(right)

        main = QVBoxLayout()
        main.addLayout(top)
        main.addSpacing(10)
        main.addWidget(QLabel("AI Output:"))
        main.addWidget(self.out)

        self.setLayout(main)

        # Bind event
        self.scan_btn.clicked.connect(self.scan)

        # Camera init
        self.cap = self.init_camera()
        self.current = None

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(30)

    def init_camera(self):
        for b in [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]:
            try:
                cam = cv2.VideoCapture(0, b)
                if cam.isOpened():
                    print("Camera opened using backend:", b)
                    return cam
            except:
                pass
        QMessageBox.critical(self, "Camera", "Unable to open camera.")
        return None

    def update_camera(self):
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)
        self.current = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
        self.cam.setPixmap(QPixmap.fromImage(q))

    # ------------------------------------------------
    # FREEZE-PROOF SCAN FUNCTION (uses threading)
    # ------------------------------------------------
    def scan(self):
        if self.current is None:
            QMessageBox.warning(self, "Wait", "Camera not ready.")
            return

        redness, bruise, swelling, preview = analyze_frame(self.current)
        pain = self.slider.value()

        # Update preview on GUI
        try:
            buf = io.BytesIO()
            preview.save(buf, format="PNG")
            q = QImage.fromData(buf.getvalue())
            self.cam.setPixmap(QPixmap.fromImage(q).scaled(640, 480, Qt.KeepAspectRatio))
        except:
            pass

        findings = (
            f"Pain level: {pain}/10\n"
            f"Redness: {redness*100:.2f}%\n"
            f"Bruise detected: {bruise}\n"
            f"Swelling estimate: {swelling:.4f}\n"
            "Give ONLY simple first-aid steps."
        )

        self.out.setText("⏳ AI analyzing injury...")

        # ---- RUN GPT IN BACKGROUND THREAD ----
        def run_gpt():
            reply = gpt_triage(preview, findings)

            def update_ui():
                self.out.setText("=== AI Triage ===\n" + reply)

            QTimer.singleShot(0, update_ui)

        threading.Thread(target=run_gpt, daemon=True).start()



def main():
    app = QApplication(sys.argv)
    w = OPDOne()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
