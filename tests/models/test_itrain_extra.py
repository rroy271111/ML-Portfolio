import sys
import runpy
import importlib
import yaml
import joblib
import types
from pathlib import Path
from unittest import mock

import pandas as pd
import numpy as np
import pytest

from models.itrain import parse_args, run, load_config

# test
def test_parse_args_default():

    ns = parse_args([])
    assert hasattr(ns, "config")
    assert ns.config == "configs/default.yaml"

def test_parse_args_custom():
    ns = parse_args(["-c", "custom_config.yaml"])
    assert ns.config == "custom_config.yaml"

# CLI invocation tests( model run)
def test_cli_exists_on_run_exception(monkeypatch, tmp_path):
    """ 
    Run the module as a script and assert it exists with code 2 when run() raises.
    monkeywatch is run to raise an exception so that the top level except block triggers sys.exit(2).
    """
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    with mock.patch("models.itrain.run", side_effect=RuntimeError("fail")):
        with pytest.raises(SystemExit) as se:
            # run module as __main__, which triggers the if __name__ == "__main__" block
            runpy.run_module("models.itrain", run_name="__main__")
        assert se.value.code == 2

def test_cli_success_run_main(monkeypatch, tmp_path):
    """
    Run the module as a script but patch run() to return normally.
    Ensure no SystemExit is raised.
    """
    # override sys.argv so argparse doesn't pick up pytest args
    monkeypatch.setattr(sys, "argv", ["itrain", "-c", "configs/default.yaml"])

    # avoid failures from outside environment
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setattr("sys.argv", ["itrain"])
    monkeypatch.setattr(
        "models.itrain.load_config",
        lambda path: {
            "model": {"trainer_path": "trainers.xgb_trainer"},
            "data": {"target_col": "label", "test_size": 0.2, "random_state": 42},
            "mlflow": {}
        }
    )

    with mock.patch("models.itrain.run", return_value={"pr_auc": 0.5}):
        # should not raise 
        result_globals = runpy.run_module("models.itrain", run_name="__main__")

        # module executed, globals returned as dict
        assert "parse_args" in result_globals
        assert "run" in result_globals