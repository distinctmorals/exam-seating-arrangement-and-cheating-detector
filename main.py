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

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create exams directory at startup if it doesn't exist
os.makedirs("exams", exist_ok=True)
app.mount("/exams", StaticFiles(directory="exams"), name="exams")

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

try:
    notes_model_path = 'my_model.pt'
    if not os.path.exists(notes_model_path):
        raise FileNotFoundError(f"Model file {notes_model_path} not found in {os.getcwd()}")
    notes_model = YOLO(notes_model_path, task='detect')
    print(f"Notes detection model loaded successfully. Class names: {notes_model.names}")
except Exception as e:
    raise Exception(f"Failed to load my_model.pt: {str(e)}")

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

def get_prev_phone_detections_map():
    if not hasattr(_thread_locals, 'prev_phone_detections_map'):
        _thread_locals.prev_phone_detections_map = {}
    return _thread_locals.prev_phone_detections_map

def get_prev_notes_detections_map():
    if not hasattr(_thread_locals, 'prev_notes_detections_map'):
        _thread_locals.prev_notes_detections_map = {}
    return _thread_locals.prev_notes_detections_map

def get_head_turn_start_map():
    if not hasattr(_thread_locals, 'head_turn_start_map'):
        _thread_locals.head_turn_start_map = {}
    return _thread_locals.head_turn_start_map

def get_low_phone_start_map():
    if not hasattr(_thread_locals, 'low_phone_start_map'):
        _thread_locals.low_phone_start_map = {}
    return _thread_locals.low_phone_start_map

# Image preprocessing for better detection
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
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

# YOLO processing function
async def batch_process_yolo(imgs, stream_ids):
    base_conf = 0.1
    phone_high_conf = 0.2
    phone_low_conf = 0.15
    notes_conf = 0.3
    processed_imgs = [preprocess_image(img) for img in imgs]
    yolo_results = await asyncio.get_event_loop().run_in_executor(
        None, lambda: yolo_model(processed_imgs, conf=base_conf, classes=[67, 73], device=device, imgsz=640, verbose=True)
    )
    notes_results = await asyncio.get_event_loop().run_in_executor(
        None, lambda: notes_model(imgs, conf=notes_conf, device=device, imgsz=640, verbose=True)
    )
    outputs = []
    prev_phone_detections = get_prev_phone_detections_map()
    prev_notes_detections = get_prev_notes_detections_map()
    low_phone_start = get_low_phone_start_map()
    for r, nr, sid in zip(yolo_results, notes_results, stream_ids):
        detected_objects = []
        all_objects = []
        phone_detected = False
        notes_detected = False
        frame_phone_detections = []
        frame_notes_detections = []
        low_conf_phone_this_frame = False
        # Process standard YOLO results (cell phone, book)
        for box in r.boxes:
            if box.cls is None:
                print(f"Stream {sid}: Empty cls detected in yolo_model, skipping box")
                continue
            cls = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
            label = yolo_model.names.get(cls, "unknown")
            conf = box.conf.item() if hasattr(box.conf, 'item') else float(box.conf)
            xyxy = box.xyxy[0].cpu().numpy().tolist() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0].tolist()
            all_objects.append(f"{label} (conf={conf:.2f}, box={xyxy})")
            if label == 'cell phone':
                if conf >= phone_high_conf:
                    phone_detected = True
                    detected_objects.append({'label': label, 'box': xyxy})
                    frame_phone_detections.append({'box': xyxy, 'conf': conf})
                elif conf >= phone_low_conf:
                    low_conf_phone_this_frame = True
                    current_time = datetime.datetime.now(pytz.UTC).timestamp()
                    if sid not in low_phone_start:
                        low_phone_start[sid] = current_time
                    else:
                        duration = current_time - low_phone_start[sid]
                        if duration >= 1:
                            phone_detected = True
                            detected_objects.append({'label': label, 'box': xyxy})
                            frame_phone_detections.append({'box': xyxy, 'conf': conf})
            elif label == 'book' and conf >= base_conf:
                detected_objects.append({'label': label, 'box': xyxy})
        # Reset low phone start if no low conf phone this frame
        if not low_conf_phone_this_frame:
            if sid in low_phone_start:
                del low_phone_start[sid]
        # Process notes detection results
        for box in nr.boxes:
            if box.cls is None:
                print(f"Stream {sid}: Empty cls detected in notes_model, skipping box")
                continue
            cls = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
            label = notes_model.names.get(cls, "unknown")
            conf = box.conf.item() if hasattr(box.conf, 'item') else float(box.conf)
            xyxy = box.xyxy[0].cpu().numpy().tolist() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0].tolist()
            print(f"Stream {sid}: Notes model detection - Class ID: {cls}, Label: {label}, Confidence: {conf:.2f}, Box: {xyxy}")
            all_objects.append(f"{label} (conf={conf:.2f}, box={xyxy})")
            if conf >= notes_conf:
                notes_detected = True
                detected_objects.append({'label': 'notes', 'box': xyxy})
                frame_notes_detections.append({'box': xyxy, 'conf': conf})
        # Temporal consistency for phone detections
        prev_phone_detections[sid] = frame_phone_detections
        if not phone_detected and prev_phone_detections.get(sid):
            for prev_box in prev_phone_detections[sid]:
                for curr_box in r.boxes:
                    if curr_box.cls is None:
                        continue
                    curr_cls = int(curr_box.cls.item()) if hasattr(curr_box.cls, 'item') else int(curr_box.cls)
                    curr_label = yolo_model.names.get(curr_cls, "unknown")
                    curr_conf = curr_box.conf.item() if hasattr(curr_box.conf, 'item') else float(curr_box.conf)
                    curr_xyxy = curr_box.xyxy[0].cpu().numpy().tolist() if hasattr(curr_box.xyxy[0], 'cpu') else curr_box.xyxy[0].tolist()
                    iou = calculate_iou(prev_box['box'], curr_xyxy)
                    if iou > 0.5 and curr_conf >= phone_low_conf * 0.8:  # Adjusted for low conf
                        detected_objects.append({'label': 'cell phone', 'box': curr_xyxy})
                        phone_detected = True
                        print(f"Stream {sid}: Phone detection recovered via temporal consistency, IoU={iou:.2f}")
        # Temporal consistency for notes detections
        prev_notes_detections[sid] = frame_notes_detections
        if not notes_detected and prev_notes_detections.get(sid):
            for prev_box in prev_notes_detections[sid]:
                for curr_box in nr.boxes:
                    if curr_box.cls is None:
                        continue
                    curr_cls = int(curr_box.cls.item()) if hasattr(curr_box.cls, 'item') else int(curr_box.cls)
                    curr_label = notes_model.names.get(curr_cls, "unknown")
                    curr_conf = curr_box.conf.item() if hasattr(curr_box.conf, 'item') else float(curr_box.conf)
                    curr_xyxy = curr_box.xyxy[0].cpu().numpy().tolist() if hasattr(curr_box.xyxy[0], 'cpu') else curr_box.xyxy[0].tolist()
                    iou = calculate_iou(prev_box['box'], curr_xyxy)
                    if iou > 0.5 and curr_conf >= notes_conf * 0.8:
                        detected_objects.append({'label': 'notes', 'box': curr_xyxy})
                        notes_detected = True
                        print(f"Stream {sid}: Notes detection recovered via temporal consistency, IoU={iou:.2f}")
        print(f"Stream {sid}: All detected objects: {all_objects}")
        detected_classes = [yolo_model.names.get(int(box.cls.item() if hasattr(box.cls, 'item') else box.cls), "unknown") for box in r.boxes if box.cls is not None] + \
                          [notes_model.names.get(int(box.cls.item() if hasattr(box.cls, 'item') else box.cls), "unknown") for box in nr.boxes if box.cls is not None]
        print(f"Stream {sid}: Detected classes: {detected_classes}")
        if not phone_detected and not notes_detected:
            print(f"Stream {sid}: No cell phone or notes detected. Check image quality, lighting, or consider fine-tuning model.")
        outputs.append((detected_objects, all_objects))
    return outputs

# Pose processing function
async def batch_process_pose(imgs, stream_ids, prev_keypoints_map, prev_head_angles_map):
    pose_results = await asyncio.get_event_loop().run_in_executor(
        None, lambda: pose_model(imgs, conf=0.2, device=device, imgsz=640)
    )
    outputs = []
    head_turn_start_map = get_head_turn_start_map()
    for pr, sid in zip(pose_results, stream_ids):
        incidents = []
        current_keypoints = []
        current_head_angles = []
        person_hands = []
        prev_keypoints = prev_keypoints_map.get(sid)
        prev_head_angles = prev_head_angles_map.get(sid, [])
        current_time = datetime.datetime.now(pytz.UTC).timestamp()
        head_turn_starts = head_turn_start_map.get(sid, {})
        current_head_turn_starts = {}
        print(f"Stream {sid}: Pose results - boxes: {bool(pr.boxes)}, keypoints: {bool(pr.keypoints)}")
        if pr and pr.boxes and pr.keypoints:
            boxes = pr.boxes
            keypoints = pr.keypoints
            print(f"Stream {sid}: Number of boxes: {len(boxes)}, Number of keypoints: {len(keypoints.xy)}")
            for idx in range(len(boxes)):
                box_conf = boxes[idx].conf.item() if hasattr(boxes[idx].conf, 'item') else boxes[idx].conf if hasattr(boxes[idx], 'conf') else boxes.conf[idx] if hasattr(boxes, 'conf') else 0.0
                if box_conf < 0.2:
                    print(f"Stream {sid}: Skipping person {idx} due to low confidence {box_conf:.2f}")
                    continue
                kp = keypoints.xy[idx].tolist() if hasattr(keypoints.xy[idx], 'tolist') else keypoints.xy[idx]
                kp_conf = keypoints.conf[idx].tolist() if hasattr(keypoints.conf[idx], 'tolist') else keypoints.conf[idx]
                current_keypoints.append(kp)
                print(f"Stream {sid}: Processing person {idx} with confidence {box_conf:.2f}")
                print(f"Stream {sid}: Keypoints for person {idx}: {kp}")
                print(f"Stream {sid}: Keypoint confidences: {kp_conf}")
                if len(kp) >= 17:
                    nose = kp[0]
                    left_ear = kp[3]
                    right_ear = kp[4]
                    left_eye = kp[1]
                    right_eye = kp[2]
                    left_shoulder = kp[5]
                    right_shoulder = kp[6]
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
                    print(f"Stream {sid}: One ear missing: {one_ear_missing}, Direction: {'right' if kp_conf[3] < 0.2 or left_ear[0] == 0 else 'left'}")
                    person_id = f"{sid}person{idx}"
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
                        if relative_angle > 40 or (one_ear_missing and missing_ear):
                            if person_id not in head_turn_starts:
                                head_turn_starts[person_id] = {'start_time': current_time, 'direction': direction}
                            elif head_turn_starts[person_id]['direction'] == direction:
                                duration = current_time - head_turn_starts[person_id]['start_time']
                                if duration >= 1:
                                    incidents.append(f"looking_at_answers_{direction}")
                                    print(f"Stream {sid}: Suspected looking at another's answers ({direction}) for person {idx}, duration={duration:.2f}s")
                            else:
                                head_turn_starts[person_id] = {'start_time': current_time, 'direction': direction}
                            current_head_turn_starts[person_id] = head_turn_starts[person_id]
                        else:
                            if person_id in head_turn_starts:
                                del head_turn_starts[person_id]
                            print(f"Stream {sid}: No significant head turn detected")
                    else:
                        if one_ear_missing:
                            direction = "right" if kp_conf[3] < 0.2 or left_ear[0] == 0 else "left"
                            if person_id not in head_turn_starts:
                                head_turn_starts[person_id] = {'start_time': current_time, 'direction': direction}
                            elif head_turn_starts[person_id]['direction'] == direction:
                                duration = current_time - head_turn_starts[person_id]['start_time']
                                if duration >= 1:
                                    incidents.append(f"looking_at_answers_{direction}")
                                    print(f"Stream {sid}: Suspected looking at another's answers ({direction}) due to missing keypoint for person {idx}, duration={duration:.2f}s")
                            else:
                                head_turn_starts[person_id] = {'start_time': current_time, 'direction': direction}
                            current_head_turn_starts[person_id] = head_turn_starts[person_id]
                        else:
                            if person_id in head_turn_starts:
                                del head_turn_starts[person_id]
                            print(f"Stream {sid}: Insufficient keypoint confidence or missing keypoints: "
                                  f"Nose conf={kp_conf[0]:.2f}, Left {'eye' if use_eyes else 'ear'} conf={left_conf:.2f}, "
                                  f"Right {'eye' if use_eyes else 'ear'} conf={right_conf:.2f}, "
                                  f"Left shoulder conf={kp_conf[5]:.2f}, Right shoulder conf={kp_conf[6]:.2f}")
                    left_hand = kp[9] if kp_conf[9] >= 0.2 else None
                    right_hand = kp[10] if kp_conf[10] >= 0.2 else None
                    person_hands.append({'left': left_hand, 'right': right_hand, 'person_idx': idx})
                    print(f"Stream {sid}: Person {idx} hand keypoints - Left wrist conf={kp_conf[9]:.2f}, Right wrist conf={kp_conf[10]:.2f}")
                else:
                    if person_id in head_turn_starts:
                        del head_turn_starts[person_id]
                head_turn_start_map[sid] = current_head_turn_starts
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
        else:
            print(f"Stream {sid}: No pose detections")
            head_turn_start_map[sid] = {}
        print(f"Stream {sid}: Pose incidents: {incidents}, Head angles: {current_head_angles}")
        outputs.append((incidents, current_keypoints, current_head_angles, []))
    return outputs

# Detect cheating endpoint
@app.post("/detect")
async def detect_cheating(data: MultiStreamData, db: Session = Depends(get_db)):
    results = []
    wat_tz = pytz.timezone('Africa/Lagos')
    prev_keypoints_map = get_prev_keypoints_map()
    prev_head_angles_map = get_prev_head_angles_map()
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
            height, width = img.shape[:2]
            new_width = 640
            new_height = int(height * (new_width / width))
            img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            rgb_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            imgs.append(rgb_img)
            stream_ids.append(stream.stream_id)
            stream_data_map[stream.stream_id] = (stream, img_resized)  # Store both stream data and resized image

        if imgs:
            yolo_outputs = await batch_process_yolo(imgs, stream_ids)
            pose_outputs = await batch_process_pose(imgs, stream_ids, prev_keypoints_map, prev_head_angles_map)

            for sid, yolo_out, pose_out in zip(stream_ids, yolo_outputs, pose_outputs):
                detected_objects, all_objects = yolo_out
                pose_incidents, current_keypoints, current_head_angles, _ = pose_out
                prev_keypoints_map[sid] = current_keypoints
                prev_head_angles_map[sid] = current_head_angles
                primary_incident = None
                if detected_objects:
                    labels = [obj['label'] for obj in detected_objects]
                    primary_incident = labels[0] if labels else None  # Use first detected object as primary
                elif pose_incidents:
                    primary_incident = pose_incidents[0]  # Use first pose incident as primary
                print(f"Stream {sid}: Status: {'suspicious' if detected_objects or pose_incidents else 'normal'}, "
                      f"Objects: {[obj['label'] for obj in detected_objects]}, Pose: {pose_incidents}, Primary: {primary_incident}")
                incidents = []
                if detected_objects:
                    incidents.append(f"objects_detected: {', '.join([obj['label'] for obj in detected_objects])}")
                incidents.extend(pose_incidents)
                stream, img_resized = stream_data_map[sid]
                # Save frame only if there are incidents
                if incidents and primary_incident:
                    current_time = datetime.datetime.now(wat_tz)
                    timestamp_str = current_time.strftime("%H-%M-%S")  # Format as HH-MM-SS
                    exam_dir = f"exams/exam_{stream.exam_id}"
                    os.makedirs(exam_dir, exist_ok=True)
                    # Use primary incident for filename, replace spaces and colons for safety
                    filename_incident = primary_incident.replace(" ", "_").replace(":", "_")
                    frame_path = f"{exam_dir}/{filename_incident}_{timestamp_str}.jpg"
                    cv2.imwrite(frame_path, img_resized)
                    print(f"Stream {sid}: Saved frame to {frame_path}")
                # Log incidents to database
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
                    "primary_incident": primary_incident or "normal",
                    "timestamp": datetime.datetime.now(wat_tz).isoformat()
                })
        db.commit()
        return results
    except Exception as e:
        print(f"Error in /detect: {str(e)}")
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