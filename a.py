from ultralytics import YOLO
import cv2
from telegram import Bot
import time
import requests
import tkinter as tk
from tkinter import filedialog
import threading

# ---------------- TELEGRAM ----------------
BOT_TOKEN = "8531599732:AAF016O9P2ekUZgAZt_QiLyZA99fqmIJdsA"
CHAT_ID = "1982984764"
bot = Bot(token=BOT_TOKEN)

# ---------------- GPS FUNCTION -------------
def get_gps_url():
    try:
        resp = requests.get("http://127.0.0.1:5000/get_gps", timeout=2).json()
        lat = resp.get("lat")
        lon = resp.get("lon")
        if lat is None or lon is None:
            return "GPS Not Available"
        return f"https://maps.google.com/?q={lat},{lon}"
    except:
        return "GPS Server Error"

# ---------------- LOAD YOUR CLASSIFICATION MODEL -------------
model = YOLO("runs/classify/train9/weights/best.pt")
CLASS_NAMES = ["Accident", "Non Accident"]

# ---------------- ALERT FUNCTION ----------------
def send_alert(frame, label, conf):
    location = get_gps_url()

    bot.send_message(
        chat_id=CHAT_ID,
        text=(
            f"🚨 Accident Confirmed!\n"
            f"Prediction: {label} ({conf:.2f})\n"
            f"Location: {location}\n"
            f"Time: {time.ctime()}"
        )
    )

    cv2.imwrite("accident.jpg", frame)
    with open("accident.jpg", "rb") as img:
        bot.send_photo(chat_id=CHAT_ID, photo=img)

# ---------------- PROCESS A SINGLE FRAME (COMMON FUNCTION) ----------------
ACCIDENT_THRESHOLD_CONF = 0.80
ACCIDENT_FRAMES_CONFIRM = 10

def classify_frame(frame, state):
    results = model(frame, imgsz=128, verbose=False)
    r = results[0]

    # RAW classification output
    top1 = int(r.probs.top1)
    top1_conf = float(r.probs.top1conf.cpu().item())
    raw_label = CLASS_NAMES[top1]

    # ---------- FIX APPLIED ----------
    if top1_conf < ACCIDENT_THRESHOLD_CONF:
        label = "Non Accident"
    else:
        label = raw_label
    # ---------------------------------

    # Display on frame
    cv2.putText(
        frame,
        f"{label} {top1_conf:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Accident confirmation logic
    if label == "Accident" and top1_conf >= ACCIDENT_THRESHOLD_CONF:
        state["frames"] += 1
    else:
        state["frames"] = 0
        state["sent"] = False

    # Send only once
    if state["frames"] >= ACCIDENT_FRAMES_CONFIRM and not state["sent"]:
        state["sent"] = True
        send_alert(frame, label, top1_conf)

    return frame

# ---------------- WEBCAM MODE ----------------
def start_webcam():
    cap = cv2.VideoCapture(0)
    state = {"frames": 0, "sent": False}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = classify_frame(frame, state)

        cv2.imshow("Accident Classifier - Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------- IMAGE MODE ----------------
def start_image():
    path = filedialog.askopenfilename(
        title="Choose Image",
        filetypes=[("Images", "*.jpg *.png *.jpeg")]
    )
    if not path:
        return

    frame = cv2.imread(path)
    state = {"frames": 10, "sent": False}  # image = instant confirm

    frame = classify_frame(frame, state)

    cv2.imshow("Accident Classifier - Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------------- VIDEO MODE ----------------
def start_video():
    path = filedialog.askopenfilename(
        title="Choose Video",
        filetypes=[("Video files", "*.mp4 *.avi")]
    )
    if not path:
        return

    cap = cv2.VideoCapture(path)
    state = {"frames": 0, "sent": False}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = classify_frame(frame, state)

        cv2.imshow("Accident Classifier - Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------- TKINTER MENU ----------------
def run_webcam_thread():
    threading.Thread(target=start_webcam).start()

def run_image_thread():
    threading.Thread(target=start_image).start()

def run_video_thread():
    threading.Thread(target=start_video).start()

root = tk.Tk()
root.title("Accident Detection - Select Mode")
root.geometry("400x400")

tk.Label(root, text="Choose Mode", font=("Arial", 18)).pack(pady=20)

tk.Button(root, text="Start Webcam", font=("Arial", 16), command=run_webcam_thread).pack(pady=15)
tk.Button(root, text="Upload Image", font=("Arial", 16), command=run_image_thread).pack(pady=15)
tk.Button(root, text="Upload Video", font=("Arial", 16), command=run_video_thread).pack(pady=15)

root.mainloop()