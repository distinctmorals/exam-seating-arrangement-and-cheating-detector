from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import datetime
import pytz
from database import SessionLocal
from models import CheatingIncident
import numpy as np
import base64
import cv2
from ultralytics import YOLO
import asyncio
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load YOLOv8n model for object detection and pose estimation
try:
    yolo_model = YOLO('yolov8n.pt')  # Nano model for object detection
    print("Object detection model loaded successfully")
except Exception as e:
    raise Exception(f"Failed to load yolov8n.pt: {str(e)}")

try:
    pose_model_path = 'yolov8n-pose.pt'
    if not os.path.exists(pose_model_path):
        raise FileNotFoundError(f"Model file {pose_model_path} not found in {os.getcwd()}")
    pose_model = YOLO(pose_model_path)  # Nano model for pose estimation
    print("Pose estimation model loaded successfully")
except Exception as e:
    raise Exception(f"Failed to load yolov8n-pose.pt: {str(e)}")

class StreamData(BaseModel):
    image: str
    user_id: str
    exam_id: str
    stream_id: str

class MultiStreamData(BaseModel):
    streams: List[StreamData]

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

async def process_yolo(rgb_img, stream_id):
    """Run YOLO detection asynchronously for phones, paper, books."""
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
    """Run pose estimation to detect communication and looking at another's answers."""
    if not stream_id:
        raise ValueError("stream_id is not defined")
    
    pose_results = await asyncio.get_event_loop().run_in_executor(
        None, lambda: pose_model(rgb_img, conf=0.3, device='cpu')
    )
    incidents = []
    current_keypoints = []
    current_head_angles = []

    # Handle pose results
    if pose_results and pose_results[0].boxes and pose_results[0].keypoints:
        # Sort boxes by confidence to prioritize the most confident person
        boxes = pose_results[0].boxes
        keypoints = pose_results[0].keypoints
        sorted_indices = np.argsort(-boxes.conf.cpu().numpy())  # Descending order
        
        # Process only the highest-confidence person
        if sorted_indices.size > 0:
            idx = sorted_indices[0]  # Index of highest-confidence person
            kp = keypoints.xy[idx].tolist()
            kp_conf = keypoints.conf[idx].tolist()
            current_keypoints.append(kp)
            print(f"Stream {stream_id}: Processing person with confidence {boxes.conf[idx]:.2f}")

            # Detect communication and looking: head orientation (using nose, ears/eyes, shoulders)
            if len(kp) >= 17:  # Standard 17-keypoint format
                nose = kp[0]  # Nose keypoint
                left_ear = kp[3]
                right_ear = kp[4]
                left_eye = kp[1]  # Left eye as fallback
                right_eye = kp[2]  # Right eye as fallback
                left_shoulder = kp[5]
                right_shoulder = kp[6]
                # Check for missing keypoints to infer head turn
                missing_ear = (kp_conf[3] < 0.3 or left_ear[0] == 0) or (kp_conf[4] < 0.3 or right_ear[0] == 0)
                one_ear_missing = (kp_conf[3] < 0.3 or left_ear[0] == 0) != (kp_conf[4] < 0.3 or right_ear[0] == 0)
                # Use ears if confident, else fallback to eyes
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
                # Check keypoint confidence and presence for angle calculation
                if (nose[0] != 0 and left_point[0] != 0 and right_point[0] != 0 and
                    left_shoulder[0] != 0 and right_shoulder[0] != 0 and
                    kp_conf[0] >= 0.3 and left_conf >= 0.3 and right_conf >= 0.3 and
                    kp_conf[5] >= 0.3 and kp_conf[6] >= 0.3):
                    # Calculate head angle relative to ears or eyes
                    head_angle = np.arctan2(right_point[1] - left_point[1], right_point[0] - left_point[0]) * 180 / np.pi
                    # Calculate shoulder angle for body orientation
                    shoulder_angle = np.arctan2(right_shoulder[1] - left_shoulder[1], right_shoulder[0] - left_shoulder[0]) * 180 / np.pi
                    # Normalize relative angle
                    relative_angle = abs((head_angle - shoulder_angle + 180) % 360 - 180)
                    current_head_angles.append(relative_angle)
                    print(f"Stream {stream_id}: Head angle: {head_angle:.2f}, Shoulder angle: {shoulder_angle:.2f}, Relative angle: {relative_angle:.2f}, Using eyes: {use_eyes}")
                    # Detect looking at another's answers (left/right head turn)
                    direction = "left" if head_angle > shoulder_angle else "right"
                    if relative_angle > 45 or (one_ear_missing and missing_ear):  # Angle or missing keypoint
                        if relative_angle > 55 or (one_ear_missing and missing_ear):  # Stronger turn or missing keypoint
                            incidents.append(f"looking_at_answers_{direction}")
                            print(f"Stream {stream_id}: Suspected looking at another's answers ({direction})")
                        else:
                            incidents.append("communication_suspected")
                            print(f"Stream {stream_id}: Communication suspected")
                    else:
                        print(f"Stream {stream_id}: No significant head turn detected")
                else:
                    # Infer looking from missing keypoints if one ear is missing
                    if one_ear_missing:
                        direction = "right" if kp_conf[3] < 0.3 or left_ear[0] == 0 else "left"
                        incidents.append(f"looking_at_answers_{direction}")
                        print(f"Stream {stream_id}: Suspected looking at another's answers ({direction}) due to missing keypoint")
                    else:
                        print(f"Stream {stream_id}: Insufficient keypoint confidence or missing keypoints: "
                              f"Nose conf={kp_conf[0]:.2f}, Left {'eye' if use_eyes else 'ear'} conf={left_conf:.2f}, "
                              f"Right {'eye' if use_eyes else 'ear'} conf={right_conf:.2f}, "
                              f"Left shoulder conf={kp_conf[5]:.2f}, Right shoulder conf={kp_conf[6]:.2f}")
            else:
                print(f"Stream {stream_id}: No valid keypoints for primary person")
        else:
            print(f"Stream {stream_id}: No person detected in pose results")
    else:
        print(f"Stream {stream_id}: No pose detections")

    # Detect paper passing: hand movement tracking
    if prev_keypoints and len(prev_keypoints) == len(current_keypoints):
        for prev_kp, curr_kp in zip(prev_keypoints, current_keypoints):
            left_hand = curr_kp[9]  # Left hand keypoint
            right_hand = curr_kp[10]  # Right hand keypoint
            prev_left = prev_kp[9]
            prev_right = prev_kp[10]
            if left_hand[0] != 0 and prev_left[0] != 0 and kp_conf[9] >= 0.3:
                left_dist = np.sqrt((left_hand[0] - prev_left[0])**2 + (left_hand[1] - prev_left[1])**2)
                if left_dist > 50:  # Significant hand movement
                    incidents.append("paper_passing_suspected")
            if right_hand[0] != 0 and prev_right[0] != 0 and kp_conf[10] >= 0.3:
                right_dist = np.sqrt((right_hand[0] - prev_right[0])**2 + (right_hand[1] - prev_right[1])**2)
                if right_dist > 50:
                    incidents.append("paper_passing_suspected")

    print(f"Stream {stream_id}: Pose incidents: {incidents}, Head angles: {current_head_angles}")
    return incidents, current_keypoints, current_head_angles

@app.post("/detect")
async def detect_cheating(data: MultiStreamData):
    db = SessionLocal()
    results = []
    wat_tz = pytz.timezone('Africa/Lagos')  # WAT is UTC+1
    prev_keypoints_map = {}  # Store keypoints for each stream
    prev_head_angles_map = {}  # Store head angles for each stream

    try:
        for stream in data.streams:
            if not stream.stream_id:
                raise HTTPException(status_code=422, detail="stream_id is missing in stream data")
            # Validate and decode base64 image
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

            # Resize image for faster processing
            img_resized = cv2.resize(img, (192, 192), interpolation=cv2.INTER_AREA)  # Reduced resolution
            rgb_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            # Object detection
            detected_objects, all_objects = await process_yolo(rgb_img, stream.stream_id)

            # Pose estimation for communication and paper passing
            pose_incidents, current_keypoints, current_head_angles = await process_pose(
                rgb_img, stream.stream_id, 
                prev_keypoints_map.get(stream.stream_id), 
                prev_head_angles_map.get(stream.stream_id, [])
            )
            prev_keypoints_map[stream.stream_id] = current_keypoints
            prev_head_angles_map[stream.stream_id] = current_head_angles

            # Determine primary incident
            primary_incident = "Normal"
            if detected_objects:
                primary_incident = f"Objects Detected: {', '.join([obj['label'] for obj in detected_objects])}"
            elif pose_incidents:
                primary_incident = f"Pose Incidents: {', '.join(pose_incidents)}"

            print(f"Stream {stream.stream_id}: Status: {'suspicious' if detected_objects or pose_incidents else 'normal'}, "
                  f"Objects: {[obj['label'] for obj in detected_objects]}, Pose: {pose_incidents}, Primary: {primary_incident}")

            # Log suspicious activities
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
    finally:
        db.close()

@app.get("/incidents/{exam_id}")
async def get_incidents(exam_id: str):
    db = SessionLocal()
    try:
        incidents = db.query(CheatingIncident).filter(CheatingIncident.exam_id == exam_id).all()
        return [{
            "id": inc.id,
            "user_id": inc.user_id,
            "stream_id": inc.stream_id,
            "incident_type": inc.incident_type,
            "timestamp": inc.timestamp.isoformat()
        } for inc in incidents]
    finally:
        db.close()