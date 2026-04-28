import json
import pytest

from src.app import create_app
from src.config import root_path


@pytest.fixture
def app():
    appl = create_app()
    appl.config.update(TESTING=True)
    root = root_path()
    model_file = root / "models" / "model.joblib"
    features = root / "configs" / "feature_columns.json"
    if (not model_file.is_file()) or \
        (not features.is_file()):
        pytest.skip("First run training (python -m src.train)")
    return appl


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def sample_payload():
    root = root_path()
    features = root / "configs" / "feature_columns.json"
    meta = json.loads(features.read_text(encoding='utf-8'))
    cols = meta['feature_columns']
    return {c: 0.0 for c in cols}