from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import pandas as pd
import uuid
from rating_systems import compute_elo


DATABASE_URL = "sqlite:///./data/newvotes.db"  # Example with SQLite, replace with PostgreSQL for production
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database model
class Vote(Base):
    __tablename__ = "votes"
    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(String, index=True)
    model_a = Column(String)
    model_b = Column(String)
    winner = Column(String)
    user_id = Column(String, index=True)
    fpath_a = Column(String)
    fpath_b = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Dependency for database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def add_vote(vote_data):
    with SessionLocal() as db:
        db_vote = Vote(**vote_data)
        db.add(db_vote)
        db.commit()
        db.refresh(db_vote)
        return {"id": db_vote.id, "user_id": db_vote.user_id, "timestamp": db_vote.timestamp}

# Function to get all votes
def get_all_votes():
    with SessionLocal() as db:
        votes = db.query(Vote).all()
        return votes

# Function to compute Elo scores
def compute_elo_scores():
    with SessionLocal() as db:
        votes = db.query(Vote).all()
        data = {
            "model_a": [vote.model_a for vote in votes],
            "model_b": [vote.model_b for vote in votes],
            "winner": [vote.winner for vote in votes]
        }
        df = pd.DataFrame(data)
        elo_scores = compute_elo(df)
        return elo_scores