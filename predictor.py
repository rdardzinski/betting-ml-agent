# predictor.py
import os
import json
import joblib
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_and_save(
    X,
    y,
    market: str,
    models_dir: str = "models",
    random_state: int = 42,
):
    """
    Trenuje model dla jednego rynku i zapisuje go na dysku.

    Args:
        X (pd.DataFrame): cechy
        y (pd.Series): target
        market (str): nazwa rynku (np. Over25)
        models_dir (str): katalog modeli

    Returns:
        float: accuracy
    """

    os.makedirs(models_dir, exist_ok=True)

    # =========================
    # WALIDACJA
    # =========================
    if X.empty:
        raise ValueError("X is empty")

    if y.nunique() < 2:
        raise ValueError(f"Target for {market} has <2 classes")

    # =========================
    # SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    # =========================
    # MODEL
    # =========================
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    # =========================
    # EWALUACJA
    # =========================
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # =========================
    # ZAPIS MODELU
    # =========================
    model_path = os.path.join(models_dir, f"{market}.joblib")
    joblib.dump(model, model_path)

    # =========================
    # METADATA
    # =========================
    metadata = {
        "market": market,
        "trained_at": datetime.utcnow().isoformat(),
        "accuracy": round(acc, 4),
        "n_samples": len(X),
        "features": list(X.columns),
    }

    meta_path = os.path.join(models_dir, f"{market}.meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return acc
