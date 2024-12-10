import os
import logging
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import pandas as pd
from datasets import load_dataset
from rating_systems import compute_elo, compute_bootstrap_elo, get_median_elo_from_bootstrap

def is_running_in_space():
    return "SPACE_ID" in os.environ

if is_running_in_space():
    DATABASE_URL = "sqlite:///./data/hf-votes.db"  
else:
    DATABASE_URL = "sqlite:///./data/local2.db"

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

def fill_database_once(dataset_name="bgsys/votes_datasets_test2"):
    with SessionLocal() as db:
        # Check if the database is already filled
        if db.query(Vote).first() is None:
            dataset = load_dataset(dataset_name)
            for record in dataset['train']:
                # Ensure the timestamp is a string
                timestamp_str = record.get("timestamp", datetime.utcnow().isoformat())
                if not isinstance(timestamp_str, str):
                    timestamp_str = datetime.utcnow().isoformat()
                
                vote_data = {
                    "image_id": record.get("image_id", ""),
                    "model_a": record.get("model_a", ""),
                    "model_b": record.get("model_b", ""),
                    "winner": record.get("winner", ""),
                    "user_id": record.get("user_id", ""),
                    "fpath_a": record.get("fpath_a", ""),
                    "fpath_b": record.get("fpath_b", ""),
                    "timestamp": datetime.fromisoformat(timestamp_str)
                }
                db_vote = Vote(**vote_data)
                db.add(db_vote)
            db.commit()
            logging.info("Database filled with data from Hugging Face dataset: %s", dataset_name)
        else:
            logging.info("Database already filled, skipping dataset loading.")

def add_vote(vote_data):
    with SessionLocal() as db:
        db_vote = Vote(**vote_data)
        db.add(db_vote)
        db.commit()
        db.refresh(db_vote)
        logging.info("Vote registered with ID: %s, using database: %s", db_vote.id, DATABASE_URL)
        return {"id": db_vote.id, "user_id": db_vote.user_id, "timestamp": db_vote.timestamp}

# Function to get all votes
def get_all_votes():
    with SessionLocal() as db:
        votes = db.query(Vote).all()
        return votes

# Function to compute Elo scores
def compute_elo_scores():
    valid_models = ["Photoroom", "RemoveBG", "BRIA RMBG 2.0"]
    
    with SessionLocal() as db:
        votes = db.query(Vote).all()
        data = {
            "model_a": [vote.model_a for vote in votes],
            "model_b": [vote.model_b for vote in votes],
            "winner": [vote.winner for vote in votes]
        }
        df = pd.DataFrame(data)
        init_size = df.shape[0]

        # Remove votes missing model_a, model_b or winner info
        df.dropna(subset=["model_a", "model_b", "winner"], inplace=True)
        
        # Validate models and winner
        def is_valid_vote(row):
            if row["model_a"] not in valid_models or row["model_b"] not in valid_models:
                return False
            if row["winner"] not in ["model_a", "model_b", "tie"]:
                return False
            return True
        
        df = df[df.apply(is_valid_vote, axis=1)]
        logging.info("Initial votes count: %d", init_size)
        logging.info("Votes count after validation: %d", df.shape[0])

        elo_scores = compute_elo(df)
        bootstrap_elo_scores = compute_bootstrap_elo(df)
        median_elo_scores = get_median_elo_from_bootstrap(bootstrap_elo_scores)
        return elo_scores, median_elo_scores

# Function to compute the number of votes for each model
def compute_votes_per_model():
    with SessionLocal() as db:
        votes = db.query(Vote).all()
        model_vote_count = {}
        
        for vote in votes:
            if vote.winner == "model_a":
                model = vote.model_a
            elif vote.winner == "model_b":
                model = vote.model_b
            else:
                continue
            
            if model not in model_vote_count:
                model_vote_count[model] = 0
            model_vote_count[model] += 1
        
        return model_vote_count
