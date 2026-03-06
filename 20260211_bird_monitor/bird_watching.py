import cv2 # type: ignore
from ultralytics import YOLO # type: ignore
import requests # type: ignore
import os
import time
from dotenv import load_dotenv # type: ignore

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
RTSP_URL = os.getenv("RTSP_URL")

# Constants
ROI_X1, ROI_Y1 = 768, 0
ROI_X2, ROI_Y2 = 2010, 1247

def send_telegram_text(text):
    """Sends a plain text message via Telegram Bot API."""
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text
    }
    try:
        response = requests.post(url, data=payload)
        # Print results into the terminal
        if response.status_code == 200:
            print("Telegram notification sent successfully!")
        else:
            print(f"Failed to send: {response.text}")
    except Exception as e:
        print(f"Notification Error: {e}")

def send_telegram_photo(photo_path, caption):
    """Sends a photo with a caption via Telegram Bot API."""
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

    # Initial system check: Send startup notification
    start_msg = "Mission Start: Monitoring the garden..."
    send_telegram_text(start_msg)
    print(start_msg)
    
    # Verify RTSP stream by capturing an initial frame
    cap.grab()
    ret, frame = cap.retrieve()
    if ret:
        preview_frame = frame.copy()
        # Display the rectangular ROI
        start_point = (ROI_X1, ROI_Y1)
        end_point = (ROI_X2, ROI_Y2)
        color = (255, 0, 0)  # BLUE (BGR)
        thickness = 5
        cv2.rectangle(preview_frame, start_point, end_point, color, thickness)
        # Put a label
        label_y = ROI_Y1 + 40 if ROI_Y1 < 50 else ROI_Y1 - 10
        cv2.putText(preview_frame, "Monitoring ROI", (ROI_X1 + 10, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        start_photo = "start_photo.jpg"
        cv2.imwrite(start_photo, preview_frame)
        start_photo_msg = "Testing Feed: Success to start monitoring."
        send_telegram_photo(start_photo, start_photo_msg)
        print(start_photo_msg)
    else:
        start_photo_msg = "Failed to start monitoring, image couldn't be captured."
        send_telegram_text(start_photo_msg)
        print(start_photo_msg)
    
    # Main monitoring loop: Analyze frames at ~3s intervals
    detected_previously = False
    while cap.isOpened():
        # Skip ~1s of frames to stay current with the live stream
        for _ in range(30):
            cap.grab()
            
        # Decode the latest frame before revision
        # ret, frame = cap.retrieve()
        # if not ret: break

        # Happy Path: Decode the latest frame
        ret, frame = cap.retrieve()

        if not ret:
            # Error Path: Implemented retry logic (10s interval) 
            # The wireless signal from a camera can be interfered with by microwave ovens.
            retry_count = 0
            max_retries = 30  # 10 x 30 = 300 [sec.] -> 5 [min.]        
            while retry_count < max_retries:
                retry_count += 1
                error_msg = f"Frame retrieval failed. Retrying... ({retry_count}/{max_retries})"
                print(error_msg)
            
                if retry_count == 1:
                    send_telegram_text("Signal disturbance detected. Initiating recovery...")
            
                cap.release()
                time.sleep(1) # Wait a second
                cap.open(RTSP_URL)
                time.sleep(9)  # Count to 10

                # Grab a few times to refresh buffer
                for _ in range(5):
                    cap.grab()
                
                ret, frame = cap.retrieve()
                if ret:
                    # Restored successfully
                    send_telegram_text(f"Connection restored after {retry_count} attempts.")
                    break # Back to Happy Path
        
        if not ret:
            # Give up and let systemd handle it
            send_telegram_text("Retries exhausted. Handing over to systemd.")
            print("Max retries reached. Exiting for systemd to take over.")
            break # Exit main loop to trigger systemd restart

        # 1. Run inference with a broad confidence threshold (0.2)
        # Targets: Person(0), Bird(14), Cat(15), Dog(16)
        roi_frame = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
        results = model.predict(roi_frame, conf=0.2, classes=[0, 14, 15, 16], verbose=False)
        boxes = results[0].boxes
        
        # 2. Filter detections based on class-specific thresholds
        found_labels = set()
        thresholds = {0: 0.6, 14: 0.3, 15: 0.4, 16: 0.4}
        names = {0: "Person", 14: "Bird", 15: "Cat", 16: "Dog"}

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            target_threshold = thresholds.get(cls_id, 0.5)
            if conf >= target_threshold:
                found_labels.add(names.get(cls_id, "Unknown"))

        # 3. Handle notifications based on detection state changes
        has_valid_target = len(found_labels) > 0

        if has_valid_target and not detected_previously:
            # Format message and save captured frame
            labels_str = ", ".join(found_labels)
            msg = f"Target confirmed: {labels_str} in the garden!"            
            photo_path = "detected_photo.jpg"
            cv2.imwrite(photo_path, frame)
            
            # Send notification and log to console
            send_telegram_photo(photo_path, msg)
            print(msg)
            
            detected_previously = True
        
        elif not has_valid_target:
            # Reset detection flag when targets leave the frame
            detected_previously = False
        
        time.sleep(2)

    cap.release()

if __name__ == "__main__":
    main()