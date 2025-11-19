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

    