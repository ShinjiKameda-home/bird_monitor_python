import cv2
from ultralytics import YOLO
import requests
import os
from dotenv import load_dotenv

# 設定の読み込み
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
RTSP_URL = os.getenv("RTSP_URL")

def send_telegram_photo(photo_path, caption):
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        payload = {"chat_id": CHAT_ID, "caption": caption}
        files = {"photo": photo}
        try:
            requests.post(url, data=payload, files=files)
        except Exception as e:
            print(f"Notification Error: {e}")

def main():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(RTSP_URL)
    print("Mission Start: Monitoring the garden...")
    
    bird_detected_previously = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model.predict(frame, conf=0.5, classes=[14], verbose=False)
        found_now = len(results[0].boxes) > 0

        if found_now and not bird_detected_previously:
            print("Target confirmed: Bird!")
            
            photo_path = "detected_bird.jpg"
            cv2.imwrite(photo_path, frame)
            
            send_telegram_photo(photo_path, "A bird has arrived in the garden!")
            
            bird_detected_previously = True
        elif not found_now:
            bird_detected_previously = False

    cap.release()

if __name__ == "__main__":
    main()