from fastapi import FastAPI, HTTPException, Depends, Request, status, Form, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import List
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

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

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
@app.post("/login", response_class=HTMLResponse)
async def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    print(f"Login: Attempting login for username '{username}'")
    db_user = db.query(User).filter(User.username == username).first()
    if not db_user or not pwd_context.verify(password, db_user.password):
        print(f"Login: Invalid credentials for username '{username}'")
        return HTMLResponse(
            content='<script>alert("Invalid username or password"); window.location="/";</script>'
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

# YOLO processing functions
async def process_yolo(rgb_img, stream_id):
    yolo_results = await asyncio.get_event_loop().run_in_executor(
        None, lambda: yolo_model(rgb_img, conf=0.3, classes=[0, 67, 73, 75], device='cpu')
    )
    detected_objects = []
    all_objects = []
    for r in yolo_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = yolo_model.names[cls]
            conf = float(box.conf)
            xyxy = box.xyxy[0].tolist()
            all_objects.append(f"{label} (conf={conf:.2f}, box={xyxy})")
            if label in ['cell phone', 'book', 'paper']:
                detected_objects.append({'label': label, 'box': xyxy})
    print(f"Stream {stream_id}: All detected objects: {all_objects}")
    return detected_objects, all_objects

async def process_pose(rgb_img, stream_id, prev_keypoints=None, prev_head_angles=None):
    if not stream_id:
        raise ValueError("stream_id is not defined")
    pose_results = await asyncio.get_event_loop().run_in_executor(
        None, lambda: pose_model(rgb_img, conf=0.3, device='cpu')
    )
    incidents = []
    current_keypoints = []
    current_head_angles = []
    person_hands = []
    if pose_results and pose_results[0].boxes and pose_results[0].keypoints:
        boxes = pose_results[0].boxes
        keypoints = pose_results[0].keypoints
        for idx in range(len(boxes)):
            if boxes.conf[idx] < 0.3:
                continue
            kp = keypoints.xy[idx].tolist()
            kp_conf = keypoints.conf[idx].tolist()
            current_keypoints.append(kp)
            print(f"Stream {stream_id}: Processing person {idx} with confidence {boxes.conf[idx]:.2f}")
            if len(kp) >= 17:
                nose = kp[0]
                left_ear = kp[3]
                right_ear = kp[4]
                left_eye = kp[1]
                right_eye = kp[2]
                left_shoulder = kp[5]
                right_shoulder = kp[6]
                missing_ear = (kp_conf[3] < 0.3 or left_ear[0] == 0) or (kp_conf[4] < 0.3 or right_ear[0] == 0)
                one_ear_missing = (kp_conf[3] < 0.3 or left_ear[0] == 0) != (kp_conf[4] < 0.3 or right_ear[0] == 0)
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
                if (nose[0] != 0 and left_point[0] != 0 and right_point[0] != 0 and
                    left_shoulder[0] != 0 and right_shoulder[0] != 0 and
                    kp_conf[0] >= 0.3 and left_conf >= 0.3 and right_conf >= 0.3 and
                    kp_conf[5] >= 0.3 and kp_conf[6] >= 0.3):
                    head_angle = np.arctan2(right_point[1] - left_point[1], right_point[0] - left_point[0]) * 180 / np.pi
                    shoulder_angle = np.arctan2(right_shoulder[1] - left_shoulder[1], right_shoulder[0] - left_shoulder[0]) * 180 / np.pi
                    relative_angle = abs((head_angle - shoulder_angle + 180) % 360 - 180)
                    current_head_angles.append(relative_angle)
                    print(f"Stream {stream_id}: Head angle: {head_angle:.2f}, Shoulder angle: {shoulder_angle:.2f}, Relative angle: {relative_angle:.2f}, Using eyes: {use_eyes}")
                    direction = "left" if head_angle > shoulder_angle else "right"
                    if relative_angle > 45 or (one_ear_missing and missing_ear):
                        if relative_angle > 55 or (one_ear_missing and missing_ear):
                            incidents.append(f"looking_at_answers_{direction}")
                            print(f"Stream {stream_id}: Suspected looking at another's answers ({direction})")
                        else:
                            incidents.append("communication_suspected")
                            print(f"Stream {stream_id}: Communication suspected")
                    else:
                        print(f"Stream_id {stream_id}: No significant head turn detected")
                else:
                    if one_ear_missing:
                        direction = "right" if kp_conf[3] < 0.3 or left_ear[0] == 0 else "left"
                        incidents.append(f"looking_at_answers_{direction}")
                        print(f"Stream {stream_id}: Suspected looking at another's answers ({direction}) due to missing keypoint")
                    else:
                        print(f"Stream {stream_id}: Insufficient keypoint confidence or missing keypoints: "
                              f"Nose conf={kp_conf[0]:.2f}, Left {'eye' if use_eyes else 'ear'} conf={left_conf:.2f}, "
                              f"Right {'eye' if use_eyes else 'ear'} conf={right_conf:.2f}, "
                              f"Left shoulder conf={kp_conf[5]:.2f}, Right shoulder conf={kp_conf[6]:.2f}")
                # Collect hands
                left_hand = kp[9] if kp_conf[9] >= 0.3 else None
                right_hand = kp[10] if kp_conf[10] >= 0.3 else None
                person_hands.append({'left': left_hand, 'right': right_hand})
            else:
                print(f"Stream {stream_id}: No valid keypoints for person {idx}")
        # Check closeness between hands of different persons
        for p1 in range(len(person_hands)):
            for p2 in range(p1 + 1, len(person_hands)):
                hands1 = [h for h in [person_hands[p1]['left'], person_hands[p1]['right']] if h]
                hands2 = [h for h in [person_hands[p2]['left'], person_hands[p2]['right']] if h]
                for h1 in hands1:
                    for h2 in hands2:
                        dist = np.sqrt((h1[0] - h2[0])**2 + (h1[1] - h2[1])**2)
                        if dist < 50:
                            incidents.append("paper_passing_suspected")
                            print(f"Stream {stream_id}: Paper passing suspected between persons {p1} and {p2}, dist {dist:.2f}")
        # Sort keypoints by nose x for matching across frames
        if current_keypoints:
            current_keypoints.sort(key=lambda kp: kp[0][0] if len(kp) > 0 else 0)
        if prev_keypoints:
            prev_keypoints.sort(key=lambda kp: kp[0][0] if len(kp) > 0 else 0)
        # Individual hand movement check
        if prev_keypoints and len(prev_keypoints) == len(current_keypoints):
            for p in range(len(current_keypoints)):
                prev_kp = prev_keypoints[p]
                curr_kp = current_keypoints[p]
                kp_conf = keypoints.conf[p].tolist() if 'conf' in keypoints.data else [0.0] * 17
                left_hand = curr_kp[9]
                right_hand = curr_kp[10]
                prev_left = prev_kp[9]
                prev_right = prev_kp[10]
                if left_hand[0] != 0 and prev_left[0] != 0 and kp_conf[9] >= 0.3:
                    left_dist = np.sqrt((left_hand[0] - prev_left[0])**2 + (left_hand[1] - prev_left[1])**2)
                    if left_dist > 50:
                        incidents.append("paper_passing_suspected")
                if right_hand[0] != 0 and prev_right[0] != 0 and kp_conf[10] >= 0.3:
                    right_dist = np.sqrt((right_hand[0] - prev_right[0])**2 + (right_hand[1] - prev_right[1])**2)
                    if right_dist > 50:
                        incidents.append("paper_passing_suspected")
    else:
        print(f"Stream {stream_id}: No pose detections")
    print(f"Stream {stream_id}: Pose incidents: {incidents}, Head angles: {current_head_angles}")
    return incidents, current_keypoints, current_head_angles

# Detect cheating endpoint
@app.post("/detect")
async def detect_cheating(data: MultiStreamData, db: Session = Depends(get_db)):
    results = []
    wat_tz = pytz.timezone('Africa/Lagos')
    prev_keypoints_map = {}
    prev_head_angles_map = {}
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
            img_resized = cv2.resize(img, (192, 192), interpolation=cv2.INTER_AREA)
            rgb_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            detected_objects, all_objects = await process_yolo(rgb_img, stream.stream_id)
            pose_incidents, current_keypoints, current_head_angles = await process_pose(
                rgb_img, stream.stream_id, 
                prev_keypoints_map.get(stream.stream_id), 
                prev_head_angles_map.get(stream.stream_id, [])
            )
            prev_keypoints_map[stream.stream_id] = current_keypoints
            prev_head_angles_map[stream.stream_id] = current_head_angles
            primary_incident = "Normal"
            if detected_objects:
                primary_incident = f"Objects Detected: {', '.join([obj['label'] for obj in detected_objects])}"
            elif pose_incidents:
                primary_incident = f"Pose Incidents: {', '.join(pose_incidents)}"
            print(f"Stream {stream.stream_id}: Status: {'suspicious' if detected_objects or pose_incidents else 'normal'}, "
                  f"Objects: {[obj['label'] for obj in detected_objects]}, Pose: {pose_incidents}, Primary: {primary_incident}")
            incidents = []
            if detected_objects:
                incidents.append(f"objects_detected: {', '.join([obj['label'] for obj in detected_objects])}")
            incidents.extend(pose_incidents)
            for incident_type in incidents:
                incident = CheatingIncident(
                    user_id=stream.user_id,
                    exam_id=stream.exam_id,
                    stream_id=stream.stream_id,
                    incident_type=incident_type,
                    timestamp=datetime.datetime.now(wat_tz)
                )
                db.add(incident)
            results.append({
                "stream_id": stream.stream_id,
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