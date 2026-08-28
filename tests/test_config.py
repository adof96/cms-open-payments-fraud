from src import config


def test_paths_are_defined():
    assert config.RAW_DATA_DIR.name == "raw"
    assert config.PROCESSED_DATA_DIR.name == "processed"
    assert config.MODELS_DIR.name == "models"
