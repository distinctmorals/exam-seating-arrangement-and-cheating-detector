from fastapi import FastAPI, HTTPException, Depends, Request, status, Form, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import List, Dict
import datetime
import uuid
from database import SessionLocal
from models import CheatingIncident, User, SessionModel
import numpy as np
import base64
import cv2
from ultralytics import YOLO
import asyncio
import os
from passlib.context import CryptContext
from sqlalchemy.orm import Session
import pytz
import pandas as pd
import io
import torch
import threading
from collections import deque

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Determine device
device = '0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLO models
try:
    yolo_model = YOLO('yolov8n.pt')
    print("Object detection model loaded successfully")
except Exception as e:
    raise Exception(f"Failed to load yolov8n.pt: {str(e)}")

try:
    pose_model_path = 'yolov8n-pose.pt'
    if not os.path.exists(pose_model_path):
        raise FileNotFoundError(f"Model file {pose_model_path} not found in {os.getcwd()}")
    pose_model = YOLO(pose_model_path)
    print("Pose estimation model loaded successfully")
except Exception as e:
    raise Exception(f"Failed to load yolov8n-pose.pt: {str(e)}")

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models
class StreamData(BaseModel):
    image: str
    user_id: str
    exam_id: str
    stream_id: str

class MultiStreamData(BaseModel):
    streams: List[StreamData]

# Get current user from session ID
async def get_current_user(request: Request, db: Session = Depends(get_db)):
    session_id = request.query_params.get('session_id')
    print(f"get_current_user: Retrieved session_id: '{session_id}'")
    if not session_id:
        print("get_current_user: No session_id provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated - no session ID provided",
            headers={"WWW-Authenticate": "Bearer"},
        )
    session = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
    print(f"get_current_user: Database query for session_id '{session_id}' returned: {session}")
    if session is None or session.expires_at < datetime.datetime.now():
        print(f"get_current_user: Invalid or expired session for session_id '{session_id}'")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = db.query(User).filter(User.id == session.user_id).first()
    print(f"get_current_user: Database query for user_id '{session.user_id}' returned: {user}")
    if user is None:
        print(f"get_current_user: User not found for session_id '{session_id}'")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found for session",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

# Signup route
@app.post("/signup", response_class=HTMLResponse)
async def signup(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    print(f"Signup: Attempting signup for username '{username}'")
    db_user = db.query(User).filter(User.username == username).first()
    if db_user:
        print(f"Signup: Username '{username}' already exists")
        return HTMLResponse(
            content='<script>alert("Username already exists"); window.location="/";</script>'
        )
    hashed_password = pwd_context.hash(password)
    new_user = User(username=username, password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    print(f"Signup: Created user '{username}' with id {new_user.id}")
    return HTMLResponse(
        content='<script>alert("Account created! Please log in."); window.location="/";</script>'
    )

# Login route
@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    print(f"Login: Attempting login for username '{username}'")
    db_user = db.query(User).filter(User.username == username).first()
    if not db_user or not pwd_context.verify(password, db_user.password):
        print(f"Login: Invalid credentials for username '{username}'")
        return HTMLResponse(
            content='<script>alert("Invalid username or password"); window.location="/";</script>',
            status_code=status.HTTP_200_OK
        )
    session_id = str(uuid.uuid4())
    expires_at = datetime.datetime.now() + datetime.timedelta(minutes=30)
    new_session = SessionModel(
        session_id=session_id,
        user_id=db_user.id,
        expires_at=expires_at
    )
    db.add(new_session)
    db.commit()
    print(f"Login: Created session '{session_id}' for user '{username}'")
    return RedirectResponse(url=f"/dashboard?session_id={session_id}", status_code=status.HTTP_303_SEE_OTHER)

# Logout route
@app.get("/logout")
async def logout(session_id: str = None, db: Session = Depends(get_db)):
    print(f"Logout: Received session_id: '{session_id}'")
    if session_id:
        session = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
        if session:
            db.delete(session)
            db.commit()
            print(f"Logout: Deleted session '{session_id}'")
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

# Dashboard route
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(current_user: User = Depends(get_current_user), session_id: str = None):
    print(f"Dashboard: Accessed by user '{current_user.username}' with session_id '{session_id}'")
    with open("static/dashboard.html", "r") as f:
        content = f.read()
    content = content.replace('{session_id}', session_id or '')
    return HTMLResponse(content=content)

# Root route
@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

# Thread-local storage for per-stream state
_thread_locals = threading.local()

def get_prev_keypoints_map():
    if not hasattr(_thread_locals, 'prev_keypoints_map'):
        _thread_locals.prev_keypoints_map = {}
    return _thread_locals.prev_keypoints_map

def get_prev_head_angles_map():
    if not hasattr(_thread_locals, 'prev_head_angles_map'):
        _thread_locals.prev_head_angles_map = {}
    return _thread_locals.prev_head_angles_map

def get_prev_looking_down_start_map():
    if not hasattr(_thread_locals, 'prev_looking_down_start_map'):
        _thread_locals.prev_looking_down_start_map = {}
    return _thread_locals.prev_looking_down_start_map

def get_frame_count_map():
    if not hasattr(_thread_locals, 'frame_count_map'):
        _thread_locals.frame_count_map = {}
    return _thread_locals.frame_count_map

def get_prev_phone_detections_map():
    if not hasattr(_thread_locals, 'prev_phone_detections_map'):
        _thread_locals.prev_phone_detections_map = {}
    return _thread_locals.prev_phone_detections_map

def get_hand_movement_history_map():
    if not hasattr(_thread_locals, 'hand_movement_history_map'):
        _thread_locals.hand_movement_history_map = {}
    return _thread_locals.hand_movement_history_map

# Image preprocessing for better phone detection
def preprocess_image(img):
    # Convert to grayscale and apply CLAHE for contrast enhancement
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Convert back to RGB
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return enhanced_rgb

# Calculate IoU for temporal consistency
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

# YOLO processing functions
async def batch_process_yolo(imgs, stream_ids):
    # Dynamic confidence threshold for cell phones
    base_conf = 0.1
    phone_conf = 0.05
    processed_imgs = [preprocess_image(img) for img in imgs]
    yolo_results = await asyncio.get_event_loop().run_in_executor(
        None, lambda: yolo_model(processed_imgs, conf=base_conf, classes=[0, 67, 73, 75], device=device, imgsz=416)
    )
    outputs = []
    prev_phone_detections = get_prev_phone_detections_map()
    for r, sid in zip(yolo_results, stream_ids):
        detected_objects = []
        all_objects = []
        phone_detected = False
        frame_phone_detections = []
        for box in r.boxes:
            if not box.cls:  # Handle empty cls
                print(f"Stream {sid}: Empty cls detected, skipping box")
                continue
            cls = int(box.cls[0])
            label = yolo_model.names.get(cls, "unknown")  # Use get to avoid mock issues
            conf = float(box.conf) if box.conf else 0.0
            # Handle both tensor and list for xyxy
            xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy[0], 'tolist') else box.xyxy[0]
            all_objects.append(f"{label} (conf={conf:.2f}, box={xyxy})")
            if label == 'cell phone' and conf >= phone_conf:
                phone_detected = True
                detected_objects.append({'label': label, 'box': xyxy})
                frame_phone_detections.append({'box': xyxy, 'conf': conf})
            elif label in ['book', 'paper'] and conf >= base_conf:
                detected_objects.append({'label': label, 'box': xyxy})
        # Temporal consistency for phone detection
        prev_detections = prev_phone_detections.get(sid, [])
        prev_phone_detections[sid] = frame_phone_detections
        if not phone_detected and prev_detections:
            for prev_box in prev_detections:
                for curr_box in r.boxes:
                    if not curr_box.cls:
                        continue
                    curr_label = yolo_model.names.get(int(curr_box.cls[0]), "unknown")
                    curr_conf = float(curr_box.conf) if curr_box.conf else 0.0
                    curr_xyxy = curr_box.xyxy[0].tolist() if hasattr(curr_box.xyxy[0], 'tolist') else curr_box.xyxy[0]
                    iou = calculate_iou(prev_box['box'], curr_xyxy)
                    if iou > 0.5 and curr_conf >= phone_conf * 0.8:
                        detected_objects.append({'label': 'cell phone', 'box': curr_xyxy})
                        phone_detected = True
                        print(f"Stream {sid}: Phone detection recovered via temporal consistency, IoU={iou:.2f}")
        print(f"Stream {sid}: All detected objects: {all_objects}")
        detected_classes = [yolo_model.names.get(int(box.cls[0]), "unknown") for box in r.boxes if box.cls] if r.boxes else []
        print(f"Stream {sid}: Detected classes: {detected_classes}")
        if not phone_detected:
            print(f"Stream {sid}: No cell phone detected. Check image quality, lighting, or consider fine-tuning model.")
        outputs.append((detected_objects, all_objects))
    return outputs

async def batch_process_pose(imgs, stream_ids, prev_keypoints_map, prev_head_angles_map, prev_looking_down_start_map):
    pose_results = await asyncio.get_event_loop().run_in_executor(
        None, lambda: pose_model(imgs, conf=0.2, device=device)
    )
    outputs = []
    hand_movement_history = get_hand_movement_history_map()
    
    for pr, sid in zip(pose_results, stream_ids):
        incidents = []
        current_keypoints = []
        current_head_angles = []
        person_hands = []
        current_looking_down_start = []
        current_time = datetime.datetime.now(pytz.UTC).timestamp()
        prev_keypoints = prev_keypoints_map.get(sid)
        prev_head_angles = prev_head_angles_map.get(sid, [])
        prev_looking_down_start = prev_looking_down_start_map.get(sid, [])
        frame_count = get_frame_count_map().get(sid, 0)
        frame_count += 1
        get_frame_count_map()[sid] = frame_count
        
        # Initialize hand movement history for this stream
        if sid not in hand_movement_history:
            hand_movement_history[sid] = {}
        
        print(f"Stream {sid}: Pose results - boxes: {bool(pr.boxes)}, keypoints: {bool(pr.keypoints)}")
        if pr and pr.boxes and pr.keypoints:
            boxes = pr.boxes
            keypoints = pr.keypoints
            print(f"Stream {sid}: Number of boxes: {len(boxes)}, Number of keypoints: {len(keypoints.xy)}")
            for idx in range(len(boxes)):
                if boxes.conf[idx] < 0.2:
                    print(f"Stream {sid}: Skipping person {idx} due to low confidence {boxes.conf[idx]:.2f}")
                    continue
                kp = keypoints.xy[idx].tolist()
                kp_conf = keypoints.conf[idx].tolist()
                current_keypoints.append(kp)
                print(f"Stream {sid}: Processing person {idx} with confidence {boxes.conf[idx]:.2f}")
                if len(kp) >= 17:
                    nose = kp[0]
                    left_ear = kp[3]
                    right_ear = kp[4]
                    left_eye = kp[1]
                    right_eye = kp[2]
                    left_shoulder = kp[5]
                    right_shoulder = kp[6]
                    neck = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2] if kp_conf[5] >= 0.2 and kp_conf[6] >= 0.2 else None
                    missing_ear = (kp_conf[3] < 0.2 or left_ear[0] == 0) or (kp_conf[4] < 0.2 or right_ear[0] == 0)
                    one_ear_missing = (kp_conf[3] < 0.2 or left_ear[0] == 0) != (kp_conf[4] < 0.2 or right_ear[0] == 0)
                    use_eyes = False
                    if missing_ear:
                        use_eyes = True
                        left_point = left_eye
                        right_point = right_eye
                        left_conf = kp_conf[1]
                        right_conf = kp_conf[2]
                    else:
                        left_point = left_ear
                        right_point = right_ear
                        left_conf = kp_conf[3]
                        right_conf = kp_conf[4]
                    print(f"Stream {sid}: Keypoint conf - Nose: {kp_conf[0]:.2f}, Left {'eye' if use_eyes else 'ear'}: {left_conf:.2f}, Right {'eye' if use_eyes else 'ear'}: {right_conf:.2f}")
                    if (nose[0] != 0 and left_point[0] != 0 and right_point[0] != 0 and
                        left_shoulder[0] != 0 and right_shoulder[0] != 0 and
                        kp_conf[0] >= 0.2 and left_conf >= 0.2 and right_conf >= 0.2 and
                        kp_conf[5] >= 0.2 and kp_conf[6] >= 0.2):
                        head_angle = np.arctan2(right_point[1] - left_point[1], right_point[0] - left_point[0]) * 180 / np.pi
                        shoulder_angle = np.arctan2(right_shoulder[1] - left_shoulder[1], right_shoulder[0] - left_shoulder[0]) * 180 / np.pi
                        relative_angle = abs((head_angle - shoulder_angle + 180) % 360 - 180)
                        current_head_angles.append(relative_angle)
                        print(f"Stream {sid}: Head angle: {head_angle:.2f}, Shoulder angle: {shoulder_angle:.2f}, Relative angle: {relative_angle:.2f}, "
                              f"Using eyes: {use_eyes}, Keypoints: nose={nose}, left_ear={left_ear}, right_ear={right_ear}, "
                              f"left_shoulder={left_shoulder}, right_shoulder={right_shoulder}")
                        direction = "left" if head_angle > shoulder_angle else "right"
                        if relative_angle > 30 or (one_ear_missing and missing_ear):
                            if relative_angle > 40 or (one_ear_missing and missing_ear):
                                incidents.append(f"looking_at_answers_{direction}")
                                print(f"Stream {sid}: Suspected looking at another's answers ({direction})")
                            else:
                                incidents.append("communication_suspected")
                                print(f"Stream {sid}: Communication suspected")
                        else:
                            print(f"Stream {sid}: No significant head turn detected")
                    else:
                        if one_ear_missing:
                            direction = "right" if kp_conf[3] < 0.2 or left_ear[0] == 0 else "left"
                            incidents.append(f"looking_at_answers_{direction}")
                            print(f"Stream {sid}: Suspected looking at another's answers ({direction}) due to missing keypoint")
                        else:
                            print(f"Stream {sid}: Insufficient keypoint confidence or missing keypoints: "
                                  f"Nose conf={kp_conf[0]:.2f}, Left {'eye' if use_eyes else 'ear'} conf={left_conf:.2f}, "
                                  f"Right {'eye' if use_eyes else 'ear'} conf={right_conf:.2f}, "
                                  f"Left shoulder conf={kp_conf[5]:.2f}, Right shoulder conf={kp_conf[6]:.2f}")
                    # Looking down detection using head tilt
                    looking_down = False
                    if neck and kp_conf[0] >= 0.2 and kp_conf[5] >= 0.2 and kp_conf[6] >= 0.2:
                        head_vector = [nose[0] - neck[0], nose[1] - neck[1]]
                        vertical_vector = [0, -1]
                        dot_product = head_vector[1] * vertical_vector[1]
                        norm_product = np.sqrt(head_vector[0]**2 + head_vector[1]**2) * np.sqrt(vertical_vector[0]**2 + vertical_vector[1]**2)
                        cos_theta = dot_product / norm_product if norm_product != 0 else 0
                        head_tilt_angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
                        if head_tilt_angle > 30 and nose[1] > neck[1]:
                            looking_down = True
                            print(f"Stream {sid}: Person {idx} looking down (head tilt angle={head_tilt_angle:.2f} degrees)")
                    if looking_down:
                        if idx < len(prev_looking_down_start) and prev_looking_down_start[idx] is None:
                            current_looking_down_start.append(current_time)
                        else:
                            current_looking_down_start.append(prev_looking_down_start[idx] if idx < len(prev_looking_down_start) else current_time)
                        start_time = current_looking_down_start[-1]
                        duration = current_time - start_time
                        if duration >= 2:
                            incidents.append("looking_down_suspected")
                            print(f"Stream {sid}: Looking down suspected for person {idx}, duration={duration:.2f}s")
                    else:
                        current_looking_down_start.append(None)
                    # Collect hands
                    left_hand = kp[9] if kp_conf[9] >= 0.2 else None
                    right_hand = kp[10] if kp_conf[10] >= 0.2 else None
                    person_hands.append({'left': left_hand, 'right': right_hand, 'person_idx': idx})
                    # Log hand keypoint confidence
                    print(f"Stream {sid}: Person {idx} hand keypoints - Left wrist conf={kp_conf[9]:.2f}, Right wrist conf={kp_conf[10]:.2f}")
                    # Check if hands are close to each other for potential paper usage (primary condition)
                    if left_hand and right_hand and left_hand[0] != 0 and right_hand[0] != 0 and kp_conf[9] >= 0.2 and kp_conf[10] >= 0.2:
                        hand_dist = np.sqrt((left_hand[0] - right_hand[0])**2 + (left_hand[1] - right_hand[1])**2)
                        print(f"Stream {sid}: Person {idx} hand distance: {hand_dist:.2f} pixels")
                        if hand_dist < 30:
                            incidents.append("potential_paper_usage")
                            print(f"Stream {sid}: Potential paper usage by person {idx}, hands close (dist={hand_dist:.2f})")
                else:
                    print(f"Stream {sid}: No valid keypoints for person {idx}")
                    current_looking_down_start.append(None)
            # Check closeness between hands of different persons for paper passing
            for p1 in range(len(person_hands)):
                for p2 in range(p1 + 1, len(person_hands)):
                    hands1 = [h for h in [person_hands[p1]['left'], person_hands[p1]['right']] if h]
                    hands2 = [h for h in [person_hands[p2]['left'], person_hands[p2]['right']] if h]
                    for h1 in hands1:
                        for h2 in hands2:
                            dist = np.sqrt((h1[0] - h2[0])**2 + (h1[1] - h2[1])**2)
                            if dist < 30:
                                incidents.append("paper_passing_suspected")
                                print(f"Stream {sid}: Paper passing suspected between persons {p1} and {p2}, dist={dist:.2f}")
            # Check single person's hand movement for potential paper usage (secondary condition)
            if prev_keypoints and len(prev_keypoints) == len(current_keypoints):
                for p in range(len(current_keypoints)):
                    person_id = f"{sid}_person_{p}"
                    if person_id not in hand_movement_history:
                        hand_movement_history[person_id] = {'left': deque(maxlen=2), 'right': deque(maxlen=2)}
                    
                    prev_kp = prev_keypoints[p]
                    curr_kp = current_keypoints[p]
                    kp_conf = keypoints.conf[p].tolist() if hasattr(keypoints, 'has_visible') and keypoints.has_visible else [0.0] * 17
                    left_hand = curr_kp[9]
                    right_hand = curr_kp[10]
                    prev_left = prev_kp[9]
                    prev_right = prev_kp[10]
                    # Check for significant hand movement
                    if left_hand[0] != 0 and prev_left[0] != 0 and kp_conf[9] >= 0.2:
                        left_dist = np.sqrt((left_hand[0] - prev_left[0])**2 + (left_hand[1] - prev_left[1])**2)
                        hand_movement_history[person_id]['left'].append(left_dist > 40)
                        if len(hand_movement_history[person_id]['left']) == 2 and all(hand_movement_history[person_id]['left']):
                            incidents.append("potential_paper_usage")
                            print(f"Stream {sid}: Potential paper usage by person {p}, left hand movement dist={left_dist:.2f}, consistent over 2 frames")
                    if right_hand[0] != 0 and prev_right[0] != 0 and kp_conf[10] >= 0.2:
                        right_dist = np.sqrt((right_hand[0] - prev_right[0])**2 + (right_hand[1] - prev_right[1])**2)
                        hand_movement_history[person_id]['right'].append(right_dist > 40)
                        if len(hand_movement_history[person_id]['right']) == 2 and all(hand_movement_history[person_id]['right']):
                            incidents.append("potential_paper_usage")
                            print(f"Stream {sid}: Potential paper usage by person {p}, right hand movement dist={right_dist:.2f}, consistent over 2 frames")
        else:
            print(f"Stream {sid}: No pose detections")
            current_looking_down_start = [None] * len(current_keypoints)
        print(f"Stream {sid}: Pose incidents: {incidents}, Head angles: {current_head_angles}")
        outputs.append((incidents, current_keypoints, current_head_angles, current_looking_down_start))
    return outputs

# Detect cheating endpoint
@app.post("/detect")
async def detect_cheating(data: MultiStreamData, db: Session = Depends(get_db)):
    results = []
    wat_tz = pytz.timezone('Africa/Lagos')
    prev_keypoints_map = get_prev_keypoints_map()
    prev_head_angles_map = get_prev_head_angles_map()
    prev_looking_down_start_map = get_prev_looking_down_start_map()
    imgs = []
    stream_ids = []
    stream_data_map = {}
    try:
        for stream in data.streams:
            if not stream.stream_id:
                raise HTTPException(status_code=422, detail="stream_id is missing in stream data")
            if not stream.image or not stream.image.startswith('data:image/jpeg;base64,'):
                raise HTTPException(status_code=422, detail=f"Invalid image data for stream {stream.stream_id}")
            try:
                img_data = base64.b64decode(stream.image.split(',')[1])
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Failed to decode image for stream {stream.stream_id}: {str(e)}")
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=422, detail=f"Invalid image format for stream {stream.stream_id}")
            # Resize to 416px width for faster processing
            height, width = img.shape[:2]
            new_width = 416
            new_height = int(height * (new_width / width))
            img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            rgb_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            imgs.append(rgb_img)
            stream_ids.append(stream.stream_id)
            stream_data_map[stream.stream_id] = stream

        if imgs:
            yolo_outputs = await batch_process_yolo(imgs, stream_ids)
            pose_outputs = await batch_process_pose(imgs, stream_ids, prev_keypoints_map, prev_head_angles_map, prev_looking_down_start_map)

            for sid, yolo_out, pose_out in zip(stream_ids, yolo_outputs, pose_outputs):
                detected_objects, all_objects = yolo_out
                pose_incidents, current_keypoints, current_head_angles, current_looking_down_start = pose_out
                prev_keypoints_map[sid] = current_keypoints
                prev_head_angles_map[sid] = current_head_angles
                prev_looking_down_start_map[sid] = current_looking_down_start
                primary_incident = "Normal"
                if detected_objects:
                    primary_incident = f"Objects Detected: {', '.join([obj['label'] for obj in detected_objects])}"
                elif pose_incidents:
                    primary_incident = f"Pose Incidents: {', '.join(pose_incidents)}"
                print(f"Stream {sid}: Status: {'suspicious' if detected_objects or pose_incidents else 'normal'}, "
                      f"Objects: {[obj['label'] for obj in detected_objects]}, Pose: {pose_incidents}, Primary: {primary_incident}")
                incidents = []
                if detected_objects:
                    incidents.append(f"objects_detected: {', '.join([obj['label'] for obj in detected_objects])}")
                incidents.extend(pose_incidents)
                stream = stream_data_map[sid]
                for incident_type in incidents:
                    incident = CheatingIncident(
                        user_id=stream.user_id,
                        exam_id=stream.exam_id,
                        stream_id=sid,
                        incident_type=incident_type,
                        timestamp=datetime.datetime.now(wat_tz)
                    )
                    db.add(incident)
                results.append({
                    "stream_id": sid,
                    "status": "suspicious" if incidents else "normal",
                    "objects": [obj['label'] for obj in detected_objects],
                    "pose_incidents": pose_incidents,
                    "primary_incident": primary_incident
                })
        db.commit()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Get incidents endpoint
@app.get("/incidents/{exam_id}")
async def get_incidents(exam_id: str, db: Session = Depends(get_db)):
    incidents = db.query(CheatingIncident).filter(CheatingIncident.exam_id == exam_id).all()
    return [{
        "id": inc.id,
        "user_id": inc.user_id,
        "stream_id": inc.stream_id,
        "incident_type": inc.incident_type,
        "timestamp": inc.timestamp.isoformat()
    } for inc in incidents]

# Generate seating endpoint
@app.post("/generate_seating")
async def generate_seating(
    file: UploadFile = File(...),
    rows: int = Form(...),
    cols: int = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    print(f"generate_seating: User '{current_user.username}' attempting to generate seating")
    try:
        content = await file.read()
        print(f"generate_seating: Received file with size {len(content)} bytes")
        df = pd.read_csv(io.BytesIO(content))
        print(f"generate_seating: CSV columns: {df.columns.tolist()}")
        required_columns = ['student_id', 'name', 'course', 'grade', 'times_cheated']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {', '.join(required_columns)}")
        
        df['student_id'] = df['student_id'].astype(int)  # Convert student_id to integer
        df['grade'] = pd.to_numeric(df['grade'], errors='coerce')
        df['times_cheated'] = pd.to_numeric(df['times_cheated'], errors='coerce')
        df['risk'] = df['times_cheated'] * 10 + (100 - df['grade']) / 10
        df = df.sort_values('risk', ascending=False)
        students = df.to_dict(orient='records')
        print(f"generate_seating: Processed {len(students)} students")
        
        grid = [[None for _ in range(cols)] for _ in range(rows)]
        
        directions_full = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        directions_adj = [(-1,0),(1,0),(0,-1),(0,1)]
        
        def get_neighbors(i, j, directions):
            nei = []
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj]:
                    nei.append(grid[ni][nj]['course'])
            return nei
        
        unplaced = []
        for stu in students:
            placed = False
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] is None:
                        neighbors = get_neighbors(i, j, directions_full)
                        if stu['course'] not in neighbors:
                            grid[i][j] = {
                                'student_id': stu['student_id'],
                                'name': stu['name'],
                                'course': stu['course'],
                                'grade': stu['grade'],
                                'times_cheated': stu['times_cheated']
                            }
                            placed = True
                            break
                if placed:
                    break
            if not placed:
                unplaced.append(stu)
        
        unplaced2 = []
        for stu in unplaced:
            placed = False
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] is None:
                        neighbors = get_neighbors(i, j, directions_adj)
                        if stu['course'] not in neighbors:
                            grid[i][j] = {
                                'student_id': stu['student_id'],
                                'name': stu['name'],
                                'course': stu['course'],
                                'grade': stu['grade'],
                                'times_cheated': stu['times_cheated']
                            }
                            placed = True
                            break
                if placed:
                    break
            if not placed:
                unplaced2.append(stu)
        
        for stu in unplaced2:
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] is None:
                        grid[i][j] = {
                            'student_id': stu['student_id'],
                            'name': stu['name'],
                            'course': stu['course'],
                            'grade': stu['grade'],
                            'times_cheated': stu['times_cheated']
                        }
                        break
        
        print(f"generate_seating: Seating arrangement generated successfully")
        return {'seating': grid}
    except Exception as e:
        print(f"generate_seating: Error - {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error generating seating: {str(e)}")

from database import Base, engine
from models import User, CheatingIncident, SessionModel

Base.metadata.create_all(bind=engine)