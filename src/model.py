"""
Загрузка и запуск модели
"""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator

from src.config import model_path, feature_columns_path


@dataclass
class ModelObject:
    est: BaseEstimator
    feature_columns: list[str]
    target_name: str
    positive_class: int


_worker: ModelObject | None = None


def load(path_model: Path | None = None, path_features: Path | None = None):
    global _worker

    if (_worker is not None) and  \
        (path_model is None) and \
        (path_features is None):
        return _worker
    
    pm = path_model
    if pm is None:
        pm = model_path()

    pf = path_features
    if pf is None:
        pf = feature_columns_path()

    if not pm.is_file():
        raise FileNotFoundError(f"Model not found: {pm}")
    if not pf.is_file():
        raise FileNotFoundError(f"Features file not found: {pf}")
    
    est = joblib.load(pm)
    metadata = json.loads(pf.read_text(encoding='utf-8'))
    features = list(metadata['feature_columns'])
    target_name = str(metadata.get('target', 'default.payment.next.month'))
    pos_class = int(metadata.get('positive_class', 1))

    m = ModelObject(
        est=est,
        feature_columns=features,
        target_name=target_name,
        positive_class=pos_class
    )

    if (path_model is None) and \
        (path_features is None):
        _worker = m

    return m


def predict(features: dict[str, any], model: ModelObject | None = None) -> tuple[int, float]:
    m = model
    if m is None:
        m = load()

    if m is None:
        raise RuntimeError("Model load is FAIL")
    # проверяем типы 
    # пытаемся привести к float
    row = []
    missing = []
    wrong_type = []
    for name in m.feature_columns:
        if name not in features:
            missing.append(name)
            continue
        val = features[name]
        if not isinstance(val, (int, float)):
            try:
                val = float(val)
            except (TypeError, ValueError):
                wrong_type.append(name)
                continue
        row.append(float(val))

    if missing:
        raise ValueError(f"Missing features: {', '.join(missing)}")
    if wrong_type:
        raise ValueError(f"Wrong values type: {', '.join(wrong_type)}")

    X = pd.DataFrame([row], columns=m.feature_columns)
    prob = m.est.predict_proba(X)[0]
    classes = list(m.est.classes_)
    try:
        idx = classes.index(m.positive_class)
    except ValueError:
        idx = 1 if len(classes) > 1 else 0
    prob_def = float(prob[idx])
    pred = int(m.est.predict(X)[0])
    return pred, prob_def