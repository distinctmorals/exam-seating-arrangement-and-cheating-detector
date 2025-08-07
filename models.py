from sqlalchemy import Column, Integer, String, DateTime
from database import Base
import datetime

class CheatingIncident(Base):
    __tablename__ = "cheating_incidents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    exam_id = Column(String, index=True)
    stream_id = Column(String, index=True)
    incident_type = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)