import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_cb(train_df, feature_cols, action_col, reward_col):
    """
    Simple one-vs-rest logistic for redemption/profit; pick argmax.
    """
    X = train_df[feature_cols].values
    y = train_df[action_col].values
    r = train_df[reward_col].values  # e.g., profit

    # Fit model predicting reward per action? Simplest: separate models per action.
    actions = np.unique(y)
    models = {}
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    for a in actions:
        mask = y == a
        # Predict P(reward>0) or regressor; for demo use logistic
        clf = LogisticRegression(max_iter=200)
        clf.fit(Xs, (r[mask] > 0).astype(int))
        models[a] = (scaler, clf)

    def policy(x_row):
        x_scaled = scaler.transform(x_row.reshape(1, -1))
        scores = {a: models[a][1].predict_proba(x_scaled)[0,1] for a in actions}
        return max(scores, key=scores.get)

    return policy, models
