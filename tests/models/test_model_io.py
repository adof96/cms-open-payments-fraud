from src.models.model_io import load_model, save_model


def test_save_model_creates_missing_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "artifact.pkl"

    save_model({"a": 1}, path)

    assert path.exists()


def test_load_model_round_trips_the_saved_object(tmp_path):
    artifacts = {"mapping": {"Psychiatry": 0.07}, "global_mean": 0.01, "feature_columns": ["x", "y"]}
    path = tmp_path / "artifacts.pkl"

    save_model(artifacts, path)

    assert load_model(path) == artifacts
