"""
opd_chatbot.py
AI OPD Triage Chatbot (Exhibition Version)
- Works on macOS without TensorFlow
- Pure OpenCV + PySimpleGUI
- Webcam injury scanning (redness, bruise-like contours)
- Pain questionnaire
- Fake detection logic
"""

import cv2
import numpy as np
import PySimpleGUI as sg
from PIL import Image, ImageTk
import io
import time
import json
from datetime import datetime

# -------------------- CONFIG --------------------
FRAME_W = 640
FRAME_H = 480
REDNESS_THRESHOLD = 0.10
CONTOUR_MIN_AREA = 2000

LOGFILE = "opd_log.jsonl"

# -------------------- HELPERS --------------------
def to_bytes(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def detect_redness(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 60, 40])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 60, 40])
    upper2 = np.array([179, 255, 255])

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)

    red_pixels = cv2.countNonZero(mask)
    total = mask.size
    fraction = red_pixels / total

    return fraction

def detect_bruise(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) > CONTOUR_MIN_AREA:
            return True
    return False

def classify_injury(red_level, bruise_flag):
    if red_level < 0.03 and not bruise_flag:
        return "No visible external injury"
    if bruise_flag and red_level > 0.08:
        return "Bruise / possible soft tissue injury"
    if red_level > 0.15:
        return "Redness detected — possible inflammation"
    return "Minor or unclear visual signs"

def advice_section(injury_type, pain_level):
    if injury_type == "No visible external injury" and pain_level >= 7:
        return "No external injury detected. Internal injuries require X-ray or clinical examination."

    if "Bruise" in injury_type:
        return "Apply ice, avoid pressure, rest the area. If pain increases, consider an X-ray."

    if "Redness" in injury_type:
        return "Likely inflammation. Rest, ice, and monitor swelling."

    return "Could not identify a clear injury. Further evaluation advised."

# -------------------- GUI --------------------
layout = [
    [sg.Text("AI OPD Triage Chatbot", font=("Arial", 22))],
    [sg.Image(key="-CAM-", size=(FRAME_W, FRAME_H))],
    [sg.Text("Pain Level (0 to 10):"), sg.Slider(range=(0,10), orientation="h", key="-PAIN-")],
    [sg.Button("Scan Injury"), sg.Button("Exit")],
    [sg.Multiline(size=(60,8), key="-OUTPUT-")]
]

window = sg.Window("OPD AI Chatbot", layout, finalize=True)

cap = cv2.VideoCapture(0)
cap.set(3, FRAME_W)
cap.set(4, FRAME_H)

# -------------------- LOOP --------------------
while True:
    event, values = window.read(timeout=20)
    if event in (sg.WIN_CLOSED, "Exit"):
        break

    ret, frame = cap.read()
    if not ret:
        continue

    window["-CAM-"].update(data=to_bytes(frame))

    if event == "Scan Injury":
        pain = int(values["-PAIN-"])

        red_level = detect_redness(frame)
        bruise_flag = detect_bruise(frame)
        inj_type = classify_injury(red_level, bruise_flag)
        advice = advice_section(inj_type, pain)

        # Fake detection
        if inj_type == "No visible external injury" and pain >= 8:
            advice = "No visible injury. Pain level is unusually high — internal issues require imaging such as X-ray."

        window["-OUTPUT-"].update(
            f"Visual Analysis: {inj_type}\n"
            f"Redness: {round(red_level*100,2)}%\n"
            f"Bruise Detected: {bruise_flag}\n"
            f"Pain Level: {pain}\n\n"
            f"Advice:\n{advice}"
        )

cap.release()
window.close()
