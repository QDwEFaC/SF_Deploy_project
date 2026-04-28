"""
Задаём пути для файлов и прочие настройки
"""

import os
from pathlib import Path


def root_path() -> Path:
    return Path(__file__).resolve().parents[1]


def model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", root_path() / "models" / "model.joblib"))


def feature_columns_path() -> Path:
    return Path(
        os.environ.get(
            "FEATURE_COLUMNS_PATH",
            root_path() / "configs" / "feature_columns.json"
        )
    )


def data_csv_path() -> Path:
    return Path(
        os.environ.get(
            "DATA_CSV_PATH",
            root_path() / "data" / "UCI_Credit_Card.csv"
        )
    )