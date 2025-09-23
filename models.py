from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)

class SessionModel(Base):
    __tablename__ = "sessions"
    session_id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    expires_at = Column(DateTime)

class CheatingIncident(Base):
    __tablename__ = "cheating_incidents"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String)
    exam_id = Column(String)
    stream_id = Column(String)
    incident_type = Column(String)
    timestamp = Column(DateTime)