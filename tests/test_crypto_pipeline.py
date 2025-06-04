import importlib
import os
import sys
import types

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest


@pytest.fixture
def graph_module(monkeypatch):
    """Load graph module with mocked langchain dependencies."""
    # Mock langchain_core.messages.HumanMessage
    lc_messages = types.ModuleType("langchain_core.messages")

    class DummyMessage:
        def __init__(self, content=None):
            self.content = content

    lc_messages.HumanMessage = DummyMessage
    monkeypatch.setitem(sys.modules, "langchain_core.messages", lc_messages)
    lc_core = types.ModuleType("langchain_core")
    lc_core.messages = lc_messages
    monkeypatch.setitem(sys.modules, "langchain_core", lc_core)

    # Mock langgraph.graph.StateGraph and END
    lg_graph = types.ModuleType("langgraph.graph")

    class DummyGraph:
        def __init__(self, *args, **kwargs):
            self.called_with = None

        def add_node(self, *a, **k):
            pass

        def add_edge(self, *a, **k):
            pass

        def set_entry_point(self, *a, **k):
            pass

        def compile(self):
            return self

        def invoke(self, data):
            self.called_with = data
            return {"result": True}

    lg_graph.StateGraph = DummyGraph
    lg_graph.END = object()
    lg = types.ModuleType("langgraph")
    lg.graph = lg_graph
    monkeypatch.setitem(sys.modules, "langgraph.graph", lg_graph)
    monkeypatch.setitem(sys.modules, "langgraph", lg)

    # Stub other imports used by the module
    dummy_mods = {
        "src.agents.portfolio_manager": types.ModuleType("src.agents.portfolio_manager"),
        "src.agents.risk_manager": types.ModuleType("src.agents.risk_manager"),
        "src.graph.state": types.ModuleType("src.graph.state"),
        "src.main": types.ModuleType("src.main"),
        "src.utils.analysts": types.ModuleType("src.utils.analysts"),
    }
    dummy_mods["src.agents.portfolio_manager"].portfolio_management_agent = lambda x: x
    dummy_mods["src.agents.risk_manager"].risk_management_agent = lambda x: x
    dummy_mods["src.graph.state"].AgentState = object
    dummy_mods["src.main"].start = lambda x: x
    dummy_mods["src.utils.analysts"].ANALYST_CONFIG = {}

    for name, mod in dummy_mods.items():
        monkeypatch.setitem(sys.modules, name, mod)

    return importlib.import_module("app.backend.services.graph")


def test_run_graph_invocation(graph_module):
    graph = graph_module.StateGraph()
    result = graph_module.run_graph(
        graph=graph,
        portfolio={"cash": 1000},
        tickers=["BTC/USDT"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        model_name="gpt",
        model_provider="OpenAI",
        exchange="binance",
    )

    assert result == {"result": True}
    assert graph.called_with["data"]["tickers"] == ["BTC/USDT"]
