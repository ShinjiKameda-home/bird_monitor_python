import cv2 # type: ignore
import numpy as np
from ultralytics import YOLO # type: ignore
import requests # type: ignore
import os
import json
from multiprocessing import shared_memory # type: ignore
import time
from dotenv import load_dotenv # type: ignore

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
RTSP_URL = os.getenv("RTSP_URL") or ""

# Static Constants
ROI_X1, ROI_Y1 = 768, 0      # The top-left point of the monitoring ROI
ROI_X2, ROI_Y2 = 2048, 1280  # The bottom-right point of the monitoring ROI
MIN_SIZE_LARGE_OBJ = 42000   # Persons, Dogs or Cats are large
MAX_SIZE_SMALL_BIRD = 12800  # Birds are small
DIFF_THRESHOLD = 28          # Sensitivity, smaller is more sensitive
MOTION_LOWER_LIMIT = 25600   # Minimum pixel sum to trigger inference
MOTION_UPPER_FACTOR = 0.8    # Max thresh to ignore Day/Night switching
FRAME_SKIP = 30              # Number of frames to grab/skip
LOOP_INTERVAL = 5            # Short sleep to prevent CPU hogging in the main loop
INFERENCE_CONF = 0.1         # Confidence parameter of the total inference
INFERENCE_CONF_PERSON = 0.65 # Confidence parameter for "Person"
INFERENCE_CONF_BIRD = 0.25   # Confidence parameter for "Bird"
INFERENCE_CONF_DOG = 0.4     # Confidence parameter for "Dog"
INFERENCE_CONF_CAT = 0.4     # Confidence parameter for "Cat"
PERMISSION_FILE = "../permission.json"  # Weather check file from Dr. Wadachi
PERM_CHECK_INTERVAL = 900               # Weather check interval [sec.]
SHEEP_COUNTING_INTERVAL = 10            # Weather check interval [sec.] when sleeping
SHM_NAME = "memories_of_haniwa_garden"  # Shared memory name for presence flag (1 byte, 0 or 1), and decision

# Global Variables
shm = None  # Shared memory object, initialized in main()

def connect_to_shm():
    """Connect to the existing shared memory created by Weather Forecast, with retry logic."""
    global shm
    while True:
        try:
            # Attempt to connect to the existing shared memory created by Weather Forecast
            shm = shared_memory.SharedMemory(name=SHM_NAME)
            break  # Successfully connected, exit the loop
        except FileNotFoundError:
            print("Waiting for Weather Forecast to create shared memory...")
            time.sleep(2)  # Wait before retrying

def update_presence(shm_obj, is_present):
    """Rewrite the zero-th byte of shared memory to indicate presence (1 for present, 0 for not present)"""
    if shm_obj is not None and shm_obj.buf is not None:
        shm_obj.buf[0] = 1 if is_present else 0

def send_telegram_text(text):
    """Send a plain text message via Telegram Bot API."""
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text
    }
    try:
        response = requests.post(url, data=payload, timeout=10)
        # Print results into the terminal
        if response.status_code == 200:
            print("Telegram notification sent successfully!")
        else:
            print(f"Failed to send: {response.text}")
    except Exception as e:
        print(f"Notification Error: {e}")

def send_telegram_photo(photo_path, caption):
    """Send a photo with a caption via Telegram Bot API."""
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        payload = {"chat_id": CHAT_ID, "caption": caption}
        files = {"photo": photo}
        try:
            requests.post(url, data=payload, files=files)
        except Exception as e:
            print(f"Notification Error: {e}")

def main():
    # Inistialize before the loop
    prev_roi_gray = None
    detected_previously = False
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(RTSP_URL)
    # Connect to global shared memory for person presence flag
    connect_to_shm() # This will wait weather_forecast.service will be running successfully.
    update_presence(shm, False) # Initialize Person presence to False.    
    send_telegram_text("Hello DrWadachi! BirdWatcher is ready to detect Persons.")

    # Send startup notification
    start_msg = "Mission Start: Monitoring the Garden..."
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

    # Initialize resting variables
    last_perm_check = 0
    is_allowed = True

    # Main monitoring loop: Analyze frames at ~3s intervals
    while cap.isOpened():
        current_time = time.time()
        check_interval = PERM_CHECK_INTERVAL if is_allowed else SHEEP_COUNTING_INTERVAL
        
        # If is_allowed, nothing will be done before PERM_CHECK_INTERVAL
        if (current_time - last_perm_check) > check_interval:
            try:
                # Check permission after the selected "INTERVAL"
                with open(PERMISSION_FILE, 'r') as f:
                    perm = json.load(f)
                    new_status = perm.get("birdwatching", True)                
                if new_status != is_allowed:
                    if not new_status:
                        send_telegram_text(f"High wind ({perm.get('wind_speed')}m/s). \n BirdWatcher is going to sleep (Zzz...)")
                    else:
                        send_telegram_text("Wind calmed down. \n BirdWatcher is waking up!")                
                is_allowed = new_status
                last_perm_check = current_time
            except Exception as e:
                last_perm_check = current_time - (check_interval - 5)
                time.sleep(SHEEP_COUNTING_INTERVAL)

        # If not is_allowed, count sheep and get back to the top of this loop
        if not is_allowed:
            time.sleep(SHEEP_COUNTING_INTERVAL)
            last_perm_check = 0
            continue

        found_labels = set()
        boxes = None

        # Skip ~1s of frames to stay current with the live stream
        for _ in range(FRAME_SKIP):
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
            time.sleep(30)
            continue # Exit main loop to trigger systemd restart

        # 0. Trim the ROI
        roi_frame = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
        current_roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        current_roi_gray = cv2.GaussianBlur(current_roi_gray, (21, 21), 0)

        motion_detected = False
        if prev_roi_gray is not None:
            # Calculate difference from the previous frame
            frame_diff = cv2.absdiff(prev_roi_gray, current_roi_gray)
            _, thresh = cv2.threshold(frame_diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
            
            # If there's any significant movement, set the flag
            diff_sum = np.sum(thresh)
            max_possible_diff = (ROI_X2 - ROI_X1) * (ROI_Y2 - ROI_Y1) * 255
            if MOTION_LOWER_LIMIT < diff_sum < (max_possible_diff * MOTION_UPPER_FACTOR):
                motion_detected = True
        
        # Update for the next loop
        prev_roi_gray = current_roi_gray.copy()

        # 1. Run inference with a broad confidence threshold (0.2)
        # Targets: Person(0), Bird(14), Cat(15), Dog(16)
        if motion_detected:
            results = model.predict(roi_frame, conf=INFERENCE_CONF, imgsz=1280, 
                                augment=True, classes=[0, 14, 15, 16], verbose=False)
            boxes = results[0].boxes
        
            # 2. Filter detections based on class-specific thresholds
            thresholds = {0: INFERENCE_CONF_PERSON, 
                          14: INFERENCE_CONF_BIRD, 
                          15: INFERENCE_CONF_CAT, 
                          16: INFERENCE_CONF_DOG}
            names = {0: "Person", 14: "Bird", 15: "Cat", 16: "Dog"}

            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    coords = box.xyxy[0].tolist()     # float coordinates
                    b_x1, b_y1, b_x2, b_y2 = map(int, coords) # integer coordinates
                    # Calculate bounding box area in pixels
                    area = (b_x2 - b_x1) * (b_y2 - b_y1)
                    # Convert coordinates from relative to global
                    gx1, gy1 = b_x1 + ROI_X1, b_y1 + ROI_Y1
                    gx2, gy2 = b_x2 + ROI_X1, b_y2 + ROI_Y1
                    # Get specific threshold for this class, defaulting to 0.5
                    target_threshold = thresholds.get(cls_id, 0.5)
                    if conf < target_threshold:
                        continue
                    # Map class ID to label name, with "Unknown" as a safety fallback
                    label_name = names.get(cls_id, "Unknown")
                    # -1. Filter out undersized 'Large' objects (e.g., wind-blown pots) ---
                    if label_name in ["Person", "Dog", "Cat"]:
                        if area < MIN_SIZE_LARGE_OBJ:
                        # Ignore small detections that are likely noise
                            continue
                    # -2. Filter out oversized 'Small' objects (e.g., large crows or misidentified cats) ---
                    if label_name == "Bird":
                        if area > MAX_SIZE_SMALL_BIRD:
                            # Only accept small-to-medium birds as "Bird"
                            continue            
                    # Draw bounding box
                    cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 0, 255), 3)
                    # Register the validated label for Telegram notification
                    found_labels.add(label_name)

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
            
            # Update shared memory to indicate presence (1 for present)
            if "Person" in found_labels or "Cat" in found_labels:
                if shm is not None and shm.buf is not None:
                    if not bool(shm.buf[0]):
                        update_presence(shm, True)  # Set presence to True 

            # Prevent repeated notifications until the target leaves the frame
            for _ in range(15): # 15s of buffer time to avoid rapid notifications
                time.sleep(1)
            detected_previously = True
        
        elif not has_valid_target:
            # Reset detection flag when targets leave the frame
            detected_previously = False
        
        time.sleep(max (0, LOOP_INTERVAL-1))

    cap.release()

if __name__ == "__main__":
    main()