import importlib
import os
import sys
import types

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest

from src.utils.parsing import parse_hedge_fund_response


@pytest.fixture
def llm_module(monkeypatch):
    """Import src.utils.llm with mocked dependencies."""
    dummy_models = types.ModuleType("src.llm.models")
    dummy_models.get_model = lambda *a, **k: None
    dummy_models.get_model_info = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "src.llm.models", dummy_models)

    dummy_progress = types.ModuleType("src.utils.progress")
    dummy_progress.progress = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "src.utils.progress", dummy_progress)

    return importlib.import_module("src.utils.llm")


def test_parse_valid_json():
    assert parse_hedge_fund_response('{"a": 1}') == {"a": 1}


def test_parse_invalid_json():
    assert parse_hedge_fund_response('{"a": 1') is None


def test_parse_wrong_type():
    assert parse_hedge_fund_response(None) is None


def test_extract_json_success(llm_module):
    content = 'some text ```json {\n  "b": 2\n} ``` end'
    assert llm_module.extract_json_from_response(content) == {"b": 2}


def test_extract_json_no_block(llm_module):
    assert llm_module.extract_json_from_response("no json here") is None


def test_extract_json_invalid_json(llm_module):
    assert llm_module.extract_json_from_response("```json {bad} ```") is None
