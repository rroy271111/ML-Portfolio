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


def test_cli_success_run_main(monkeypatch):
    """
    Test the itrain module from CLI
    """

    # create a fake trainer module that looks like src.trainers.xgb_trainer
    # we avoid importing the real one, because the CLI would try to load it.
    # SimpleNamespace acts like an object with attributes.
    # when itrain calls trainer.train(...) this fake one will be used there
    fake_trainer = types.SimpleNamespace(
        train=lambda *a, **k: ("model", {"pr_auc": 0.5})
    )

    # put fake trainer into sys.modules before the import happens.
    # importlib.import checks sys.modules first
    # so this prevents ModuleNotFoundError inside models.itrain
    monkeypatch.setitem(sys.modules, "src.trainers.xgb_trainer", fake_trainer)

    # monkeypatch.context() ensures patches only exist inside this block.
    # when the block exits, patches are undone.
    with monkeypatch.context() as m:

        # avoid the actual CLI from running exit() or reading pytest args
        # the app checks this variable in __main__
        m.setenv("PYTEST_RUNNING", "1")

        # fake sys.argv so argparse doesn't try to parse pytest args
        # runpy.run_module acts like invoking the script
        m.setattr("sys.argv", ["itrain"])

        # monkeypatch parse_args() so the CLI thinks we passed -c config
        # no argparse is executed here, we directly specify return value
        m.setattr(
            "models.itrain.parse_args",
            lambda: mock.Mock(config="configs/default.yaml"),
            raising=False,
        )
        # run the module as if we executed:
        #  python -m models.itrain
        # runpy loads the module fresh into a new globals dict,
        # executing the __main__ block.
        result_globals = runpy.run_module("models.itrain", run_name="__main__")

    # runpy.run_module returns a dict containing all globals in the module.
    assert "run" in result_globals
    assert "parse_args" in result_globals
