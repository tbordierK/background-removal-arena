# This code is copied from the following source:
# https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/monitor/rating_systems.py

import math
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
import pandas as pd

def get_matchups_models(df):
    n_rows = len(df)
    model_indices, models = pd.factorize(pd.concat([df["model_a"], df["model_b"]]))
    matchups = np.column_stack([model_indices[:n_rows], model_indices[n_rows:]])
    return matchups, models.to_list()


def preprocess_for_elo(df):
    """
    in Elo we want numpy arrays for matchups and outcomes
      matchups: int32 (N,2)  contains model ids for the competitors in a match
      outcomes: float64 (N,) contains 1.0, 0.5, or 0.0 representing win, tie, or loss for model_a
    """
    matchups, models = get_matchups_models(df)
    outcomes = np.full(len(df), 0.5)
    outcomes[df["winner"] == "model_a"] = 1.0
    outcomes[df["winner"] == "model_b"] = 0.0
    return matchups, outcomes, models


def compute_elo(df, k=4.0, base=10.0, init_rating=1000.0, scale=400.0):
    matchups, outcomes, models = preprocess_for_elo(df)
    alpha = math.log(base) / scale
    ratings = np.full(shape=(len(models),), fill_value=init_rating)
    for (model_a_idx, model_b_idx), outcome in zip(matchups, outcomes):
        prob = 1.0 / (
            1.0 + math.exp(alpha * (ratings[model_b_idx] - ratings[model_a_idx]))
        )
        update = k * (outcome - prob)
        ratings[model_a_idx] += update
        ratings[model_b_idx] -= update
    return {model: ratings[idx] for idx, model in enumerate(models)}


def compute_elo_from_votes(db: Session):
    # Retrieve all votes from the database
    votes = db.query(Vote).all()
    
    # Convert votes to a DataFrame
    data = {
        "model_a": [vote.model_a for vote in votes],
        "model_b": [vote.model_b for vote in votes],
        "winner": [vote.winner for vote in votes]
    }
    df = pd.DataFrame(data)
    
    # Compute Elo scores using the existing function
    elo_scores = compute_elo(df)
    
    return elo_scores