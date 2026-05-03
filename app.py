from flask import Flask, render_template, request, Response, jsonify
from ultralytics import YOLO
import cv2
import time
from telegram import Bot
import requests
import asyncio

app = Flask(__name__)

# ---- Telegram Alert ----
BOT_TOKEN = "8531599732:AAF016O9P2ekUZgAZt_QiLyZA99fqmIJdsA"
CHAT_ID = "1982984764"

model = YOLO("runs/classify/train9/weights/best.pt")
CLASS_NAMES = ["Accident", "Non Accident"]

ACCIDENT_THRESHOLD_CONF = 0.80
ACCIDENT_FRAMES = 10

# Shared state for current video label
current_label = {"label": "Processing...", "conf": 0.0}


# ---------------- FIXED LOCATION (REVA UNIVERSITY) ----------------
def get_gps_url():
    lat = 13.1168745
    lon = 77.6346118
    return f"https://maps.google.com/?q={lat},{lon}"
# -----------------------------------------------------------------


async def _send_alert_async(frame, label, conf):
    location = get_gps_url()
    async with Bot(token=BOT_TOKEN) as b:
        await b.send_message(
            chat_id=CHAT_ID,
            text=f"🚨 Accident Detected!\nPrediction: {label} ({conf:.2f})\nLocation: {location}\nTime: {time.ctime()}"
        )
        cv2.imwrite("accident.jpg", frame)
        with open("accident.jpg", "rb") as f:
            await b.send_photo(chat_id=CHAT_ID, photo=f)


def send_alert(frame, label, conf):
    asyncio.run(_send_alert_async(frame, label, conf))


def classify_frame(frame, state):
    result = model(frame, imgsz=160, verbose=False)[0]

    top1 = int(result.probs.top1)
    conf = float(result.probs.top1conf.cpu().item())
    label = CLASS_NAMES[top1]

    if conf < ACCIDENT_THRESHOLD_CONF:
        label = "Non Accident"

    current_label["label"] = label
    current_label["conf"] = round(conf, 2)

    if label == "Accident" and conf >= ACCIDENT_THRESHOLD_CONF:
        state["frames"] += 1
    else:
        state["frames"] = 0
        state["sent"] = False

    if state["frames"] >= ACCIDENT_FRAMES and not state["sent"]:
        state["sent"] = True
        send_alert(frame, label, conf)

    cv2.putText(
        frame, 
        f"{label} {conf:.2f}", 
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 0, 255), 
        2
    )
    return frame


# ------------------- ROUTES --------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/webcam")
def webcam():
    return render_template("webcam.html")


def webcam_stream():
    cap = cv2.VideoCapture(0)
    state = {"frames": 0, "sent": False}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = classify_frame(frame, state)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/webcam_feed")
def webcam_feed():
    return Response(
        webcam_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/upload_video", methods=["POST"])
def upload_video():
    file = request.files["video"]
    path = "uploaded_video.mp4"
    file.save(path)
    return jsonify({"status": "ok", "path": "/play_video"})


@app.route("/play_video")
def play_video_page():
    return render_template("video_page.html")


def video_stream():
    cap = cv2.VideoCapture("uploaded_video.mp4")
    state = {"frames": 0, "sent": False}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = classify_frame(frame, state)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(
        video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/video_label")
def video_label():
    return jsonify(current_label)


if __name__ == "__main__":
    app.run(debug=True)