"""Checks for the three notebooks, saved models and vehicle separation."""
import ast
import copy
import hashlib
import json
import sys
import joblib
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ["01_understanding_the_data.ipynb", "02_comparing_the_models.ipynb", "03_improving_and_testing.ipynb"]


def sources(name):
    return ["".join(cell["source"]) for cell in json.loads((ROOT / "notebooks" / name).read_text())["cells"]
            if cell["cell_type"] == "code"]


@pytest.fixture
def study(tmp_path):
    import shutil
    for folder in ["data", "results/tables", "results/predictions", "models/baseline"]:
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    for name in ["synthetic_telemetry_data.xlsx", "vehicle_folds.csv"]:
        shutil.copy2(ROOT / "data" / name, tmp_path / "data" / name)
    code = sources(NOTEBOOKS[1])
    setup = ast.parse(code[0])
    packages = next(ast.literal_eval(node.value) for node in setup.body
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "packages")
    namespace = {"project": tmp_path, "tables": tmp_path / "results/tables", "Path": Path,
                 "hashlib": hashlib, "json": json, "np": np, "pd": pd, "version": version,
                 "packages": packages, "sys": sys, "joblib": joblib}
    for part in code[1:]:
        if part.startswith(("data_path =", "import time", "def grouped_splits", "model_settings =",
                            "def prepare_inputs", "FIT_COUNT =", "data_signature =")):
            exec(compile(part, "<notebook>", "exec"), namespace)
    return namespace


def test_three_notebooks_are_valid_python():
    assert sorted(path.name for path in (ROOT / "notebooks").glob("*.ipynb")) == NOTEBOOKS
    for name in NOTEBOOKS:
        for part in sources(name):
            ast.parse(part)
        text = (ROOT / "notebooks" / name).read_text().lower()
        assert "scope control" not in text
        assert "<svg" not in text


def test_groups_and_features(study):
    frame = study["df"]
    assert len(study["numeric_columns"]) == 32
    assert len(study["predictors"]) == 33
    assert set(study["predictors"]).isdisjoint(study["excluded"] + ["abs_fault_indicator"])
    for fold in range(1, 6):
        test = frame.loc[study["fold_for_row"].eq(fold)]
        train = frame.loc[~study["fold_for_row"].eq(fold)]
        assert test.vehicle_id.nunique() == 10 and test[study["target"]].sum() == 7
        assert set(test.vehicle_id).isdisjoint(train.vehicle_id)
        for a, b in study["grouped_splits"](train):
            assert set(train.iloc[a].vehicle_id).isdisjoint(train.iloc[b].vehicle_id)


def test_cached_replay_never_fits(study):
    config = study["experiment"]("LR")
    config["candidates"] = config["candidates"][:1]
    first_scores, first_predictions = study["run_experiment"](config)
    fits = study["FIT_COUNT"]
    assert fits == 20
    def unexpected_training(*args, **kwargs):
        raise AssertionError("Cached replay attempted training")
    study["fit_model"] = unexpected_training
    second_scores, second_predictions = study["run_experiment"](config)
    assert study["FIT_COUNT"] == fits
    assert first_scores == second_scores
    pd.testing.assert_frame_equal(first_predictions, second_predictions)
    assert first_scores["TP"] + first_scores["FN"] == 35
    assert first_scores["TN"] + first_scores["FP"] == 1935


def test_cache_settings_are_checked(study):
    config = study["experiment"]("LR")
    config["candidates"] = config["candidates"][:1]
    study["run_experiment"](config)
    changed = copy.deepcopy(config)
    changed["candidates"][0]["params"]["C"] = 999
    with pytest.raises(ValueError, match="different study settings"):
        study["run_experiment"](changed)


def test_original_sampling_rows_are_preserved(study):
    part = next(source for source in sources(NOTEBOOKS[2]) if source.startswith("def add_training_examples"))
    exec(part, study)
    train = study["df"].loc[~study["fold_for_row"].eq(1)]
    for method in ["random_over", "smotenc"]:
        config = study["experiment"]("LR", sampler=method, weighted=False)
        values, labels = study["add_training_examples"](train, config, .1)
        pd.testing.assert_frame_equal(values.iloc[:len(train)].reset_index(drop=True),
                                      train[study["predictors"]].reset_index(drop=True))
        np.testing.assert_array_equal(labels[:len(train)], train[study["target"]])
        assert len(labels) - len(train) == int(np.ceil(train[study["target"]].eq(0).sum() * .1)) - 28
