import pytest
import asyncio
from fastapi.testclient import TestClient
from main import app, get_db, pwd_context, yolo_model, pose_model
from database import SessionLocal, Base, engine
from models import User, SessionModel, CheatingIncident
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import datetime
import base64
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
import pandas as pd
import io
import python_multipart

# Set up a test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
test_engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Override get_db dependency for testing
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

# Create test database tables
Base.metadata.create_all(bind=test_engine)

client = TestClient(app, follow_redirects=False)  # Disable redirect following

@pytest.fixture(autouse=True)
def setup_and_teardown():
    # Setup: Clear database
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)
    yield
    # Teardown: Clean up
    Base.metadata.drop_all(bind=test_engine)

@pytest.fixture
def test_user():
    db = TestingSessionLocal()
    hashed_password = pwd_context.hash("testpassword")
    user = User(username="testuser", password=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)
    yield user
    db.close()

@pytest.fixture
def test_session(test_user):
    db = TestingSessionLocal()
    session_id = "test_session_id"
    session = SessionModel(
        session_id=session_id,
        user_id=test_user.id,
        expires_at=datetime.datetime.now() + datetime.timedelta(minutes=30)
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    yield session
    db.close()

# TC-U-01: Verify user login
def test_user_login(test_user):
    response = client.post("/login", data={"username": "testuser", "password": "testpassword"})
    assert response.status_code == 303  # Redirect to dashboard
    assert "session_id" in response.headers["location"]
    assert response.headers["location"].startswith("/dashboard?session_id=")

# TC-U-02: Validate head pose anomaly
@patch("main.pose_model")
def test_pose_detection(mock_pose_model):
    mock_result = MagicMock()
    mock_keypoints = MagicMock()
    mock_keypoints.xy = [[
        [100, 50],  # nose
        [90, 50],   # left_eye
        [110, 50],  # right_eye
        [80, 50],   # left_ear
        [200, 80],  # right_ear
        [90, 100],  # left_shoulder
        [110, 100], # right_shoulder
        [0, 0], [0, 0], [0, 0], [0, 0],  # placeholders
        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]
    ]]
    mock_keypoints.conf = [[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    mock_box = MagicMock(conf=0.9)
    mock_result.boxes = [mock_box]
    mock_result.keypoints = mock_keypoints
    mock_pose_model.return_value = [mock_result]
    
    img = np.zeros((416, 416, 3), dtype=np.uint8)
    stream_id = "test_stream"
    
    from main import batch_process_pose, get_prev_keypoints_map, get_prev_head_angles_map, get_prev_looking_down_start_map
    result = asyncio.run(batch_process_pose([img], [stream_id], get_prev_keypoints_map(), get_prev_head_angles_map(), get_prev_looking_down_start_map()))
    incidents = result[0][0]
    assert "looking_at_answers_right" in incidents or "communication_suspected" in incidents

# TC-U-03: Detect unauthorized phone
@patch("main.yolo_model")
def test_object_detection(mock_yolo_model):
    mock_result = MagicMock()
    mock_box = MagicMock()
    mock_box.cls = [67]  # Cell phone class
    mock_box.conf = 0.6  # Scalar confidence
    mock_box.xyxy = [[100, 100, 150, 150]]  # Plain list for coordinates
    mock_result.boxes = [mock_box]
    mock_result.names = {67: "cell phone"}
    mock_yolo_model.return_value = [mock_result]
    mock_yolo_model.names = {67: "cell phone"}  # Direct dictionary for names
    
    img = np.zeros((416, 416, 3), dtype=np.uint8)
    stream_id = "test_stream"
    
    from main import batch_process_yolo
    result = asyncio.run(batch_process_yolo([img], [stream_id]))
    detected_objects = result[0][0]
    assert any(obj["label"] == "cell phone" for obj in detected_objects)

# TC-I-01: Process video and trigger alert
@pytest.mark.asyncio
async def test_video_to_alert(test_session):
    # Mock video frame with a phone
    img = np.zeros((416, 416, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()
    
    with patch("main.yolo_model") as mock_yolo:
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.cls = [67]
        mock_box.conf = 0.6  # Scalar confidence
        mock_box.xyxy = [[100, 100, 150, 150]]  # Plain list
        mock_result.boxes = [mock_box]
        mock_result.names = {67: "cell phone"}
        mock_yolo.return_value = [mock_result]
        mock_yolo.names = {67: "cell phone"}  # Direct dictionary for names
        
        response = client.post(
            "/detect",
            json={
                "streams": [{
                    "image": img_b64,
                    "user_id": str(test_session.user_id),
                    "exam_id": "test_exam",
                    "stream_id": "test_stream"
                }]
            }
        )
    
    assert response.status_code == 200
    results = response.json()
    assert results[0]["status"] == "suspicious"
    assert "cell phone" in results[0]["objects"]

# TC-I-02: Assign seat and update database
@pytest.mark.asyncio
async def test_seating_to_database(test_session):
    csv_content = "student_id,name,course,grade,times_cheated\n1,John Doe,Math,85,0"
    csv_file = io.BytesIO(csv_content.encode())
    csv_file.name = "test.csv"
    
    response = client.post(
        f"/generate_seating?session_id={test_session.session_id}",
        files={"file": ("test.csv", csv_file, "text/csv")},
        data={"rows": "1", "cols": "1"}
    )
    
    assert response.status_code == 200
    seating = response.json()["seating"]
    assert len(seating) == 1
    assert seating[0][0]["student_id"] == 1  # Expect integer as per main.py

# TC-S-01: Full exam workflow
@pytest.mark.asyncio
async def test_end_to_end_workflow(test_user, test_session):
    # Step 1: Login
    response = client.post("/login", data={"username": "testuser", "password": "testpassword"})
    assert response.status_code == 303
    
    # Step 2: Generate seating
    csv_content = "student_id,name,course,grade,times_cheated\n1,John Doe,Math,85,0"
    csv_file = io.BytesIO(csv_content.encode())
    csv_file.name = "test.csv"
    response = client.post(
        f"/generate_seating?session_id={test_session.session_id}",
        files={"file": ("test.csv", csv_file, "text/csv")},
        data={"rows": "1", "cols": "1"}
    )
    assert response.status_code == 200
    
    # Step 3: Detect cheating
    img = np.zeros((416, 416, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()
    
    with patch("main.yolo_model") as mock_yolo:
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.cls = [67]
        mock_box.conf = 0.6  # Scalar confidence
        mock_box.xyxy = [[100, 100, 150, 150]]  # Plain list
        mock_result.boxes = [mock_box]
        mock_result.names = {67: "cell phone"}
        mock_yolo.return_value = [mock_result]
        mock_yolo.names = {67: "cell phone"}  # Direct dictionary for names
        
        response = client.post(
            "/detect",
            json={
                "streams": [{
                    "image": img_b64,
                    "user_id": str(test_session.user_id),
                    "exam_id": "test_exam",
                    "stream_id": "test_stream"
                }]
            }
        )
    
    assert response.status_code == 200
    results = response.json()
    assert results[0]["status"] == "suspicious"
    
    # Step 4: Verify incidents in database
    response = client.get("/incidents/test_exam")
    assert response.status_code == 200
    incidents = response.json()
    assert len(incidents) > 0
    assert incidents[0]["incident_type"].startswith("objects_detected")

# TC-S-02: Handle 20 camera feeds
@pytest.mark.asyncio
async def test_performance_20_streams(test_session):
    streams = [
        {
            "image": "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", np.zeros((416, 416, 3), dtype=np.uint8))[1]).decode(),
            "user_id": str(test_session.user_id),
            "exam_id": "test_exam",
            "stream_id": f"stream_{i}"
        } for i in range(20)
    ]
    
    with patch("main.yolo_model") as mock_yolo, patch("main.pose_model") as mock_pose:
        # YOLO mock for object detection
        mock_yolo_result = MagicMock()
        mock_yolo_box = MagicMock()
        mock_yolo_box.cls = [67]  # Cell phone class
        mock_yolo_box.conf = 0.6
        mock_yolo_box.xyxy = [[100, 100, 150, 150]]
        mock_yolo_result.boxes = [mock_yolo_box]
        mock_yolo_result.names = {67: "cell phone"}
        mock_yolo.return_value = [mock_yolo_result] * 20
        mock_yolo.names = {67: "cell phone"}
        
        # Pose mock for pose detection
        mock_pose_result = MagicMock()
        mock_pose_keypoints = MagicMock()
        mock_pose_keypoints.xy = [[
            [100, 50], [90, 50], [110, 50], [80, 50], [200, 80],
            [90, 100], [110, 100], [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]
        ]]
        mock_pose_keypoints.conf = [[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        mock_pose_box = MagicMock(conf=0.9)
        mock_pose_result.boxes = [mock_pose_box]
        mock_pose_result.keypoints = mock_pose_keypoints
        mock_pose.return_value = [mock_pose_result] * 20
        
        response = client.post("/detect", json={"streams": streams})
    
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 20
    assert all(result["status"] == "suspicious" for result in results)  # Expect suspicious due to cell phone

# TC-S-03: Prevent SQL injection
def test_sql_injection(test_user):
    malicious_input = "testuser'; DROP TABLE users; --"
    response = client.post("/login", data={"username": malicious_input, "password": "testpassword"})
    assert response.status_code == 200  # Returns HTML response with error
    assert "Invalid username or password" in response.text
    
    # Verify database integrity
    db = TestingSessionLocal()
    users = db.query(User).all()
    db.close()
    assert len(users) > 0  # Ensure users table still exists

if __name__ == "__main__":
    pytest.main(["-v", __file__])