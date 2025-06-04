import os
import sys
from unittest.mock import patch

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pytest.importorskip("langchain_core", reason="langgraph stack not installed")

from app.backend.services.graph import create_graph, run_graph
from app.backend.services.portfolio import create_portfolio
from src.utils.llm import create_default_response
import src.utils.llm as llm


def _fake_call_llm(*args, **kwargs):
    pydantic_model = kwargs.get("pydantic_model")
    default_factory = kwargs.get("default_factory")
    if default_factory:
        return default_factory()
    return create_default_response(pydantic_model)


def test_crypto_smoke():
    with patch.object(llm, "call_llm", side_effect=_fake_call_llm):
        graph = create_graph(["crypto_sentiment_analyst"])
        agent = graph.compile()
        portfolio = create_portfolio(1000.0, 0.0, ["BTC/USDT"])
        result = run_graph(
            graph=agent,
            portfolio=portfolio,
            tickers=["BTC/USDT"],
            start_date="2024-01-01",
            end_date="2024-01-02",
            model_name="fake",
            model_provider="OpenAI",
            exchange="binance",
        )
        assert result is not None
