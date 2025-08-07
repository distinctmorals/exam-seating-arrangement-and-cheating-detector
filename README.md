Exam Cheating Detector
Setup Instructions

Create a project directory and ensure all files are placed in the correct structure (with static/ folder for HTML, CSS, and JS).
Install dependencies:

pip install -r requirements.txt


Create the SQLite database tables:

from database import Base, engine
Base.metadata.create_all(bind=engine)


Download the YOLOv8 model weights (yolov8n.pt) from the Ultralytics website and place it in the project directory.
Run the application:

uvicorn main:app --reload


Access the application at http://localhost:8000

Features

Monitors multiple video streams simultaneously
Detects faces, phones, papers, and talking
Uses SQLite for storing cheating incidents
Displays live status for each stream and overall status
Logs incidents with timestamps, user IDs, and stream IDs
Supports detection of:
Multiple faces or no faces
Phones and papers using YOLOv8
Talking through audio analysis



Notes

Ensure cameras and microphones are connected
The system checks streams every 2 seconds
Incidents are updated every 5 seconds
Audio detection is basic and may need calibration
YOLOv8 model requires the yolov8n.pt weights file
Modify the streams array in script.js to add more students/streams
