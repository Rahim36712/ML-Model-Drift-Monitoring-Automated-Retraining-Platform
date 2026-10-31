import pytest
from src.config.settings import get_settings
from src.data.database import get_database
from src.runtime import management_token
from src.pipeline.deployer import ModelProvider


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("MLOPS_HOME", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLOPS_DATASET_PATH", raising=False)
    get_settings.cache_clear()
    get_database.cache_clear()
    management_token.cache_clear()
    ModelProvider.set_active_model(None, None, None)
    settings = get_settings()
    settings.model_params.n_estimators = 25
    db = get_database(settings.database.url)
    db.init_db()
    yield settings, db
    ModelProvider.set_active_model(None, None, None)
    db.engine.dispose()
    get_database.cache_clear()
    get_settings.cache_clear()
    management_token.cache_clear()
