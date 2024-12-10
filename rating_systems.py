# This code is copied from the following source:
# https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/monitor/rating_systems.py

import math
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
import pandas as pd
from scipy.special import expit

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



def compute_bootstrap_elo(
    df, num_round=100, k=4.0, base=10.0, init_rating=1000.0, scale=400.0
):
    matchups, outcomes, models = preprocess_for_elo(df)
    sample_indices = np.random.randint(low=0, high=len(df), size=(len(df), num_round))
    ratings = fit_vectorized_elo(
        matchups, outcomes, sample_indices, len(models), k, base, init_rating, scale
    )
    df = pd.DataFrame(data=ratings, columns=models)
    return df[df.median().sort_values(ascending=False).index]

def fit_vectorized_elo(
    matchups,
    outcomes,
    sample_indices,
    num_models,
    k=4.0,
    base=10.0,
    init_rating=1000.0,
    scale=400.0,
):
    """fit multiple sets of Elo ratings on different samples of the data at the same time"""
    alpha = math.log(base) / scale
    num_samples = sample_indices.shape[1]
    ratings = np.zeros(shape=(num_samples, num_models), dtype=np.float64)
    # iterate over the rows of sample_indices, each column is an index into a match in the input arrays
    sample_range = np.arange(num_samples)
    for matchup_indices in sample_indices:
        model_a_indices = matchups[matchup_indices, 0]
        model_b_indices = matchups[matchup_indices, 1]
        model_a_ratings = ratings[sample_range, model_a_indices]
        model_b_ratings = ratings[sample_range, model_b_indices]
        sample_outcomes = outcomes[matchup_indices]
        probs = expit(alpha * (model_a_ratings - model_b_ratings))
        updates = k * (sample_outcomes - probs)
        ratings[sample_range, model_a_indices] += updates
        ratings[sample_range, model_b_indices] -= updates
    return ratings + init_rating

def get_median_elo_from_bootstrap(bootstrap_df):
    median = dict(bootstrap_df.quantile(0.5))
    median = {k: int(v + 0.5) for k, v in median.items()}
    return median