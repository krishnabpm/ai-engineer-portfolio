"""Tests for the chatbot. They use a fake model, so no network or API key is needed."""
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import openrouter_streaming_chatbot as bot


def chunks(*texts):
    """Build fake streamed chunks (objects with a .content attribute)."""
    return [SimpleNamespace(content=t) for t in texts]


class FakeModel:
    """Stands in for ChatOpenAI. Each stream() call returns the next scripted reply.

    A scripted item can be a list of chunks, or an Exception to raise.
    Every call records a copy of the messages it was given.
    """

    def __init__(self, *scripted):
        self.scripted = list(scripted)
        self.calls = []

    def stream(self, messages):
        self.calls.append(list(messages))
        item = self.scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        return iter(item)


def run_chat(model, *user_inputs):
    """Run chat_loop feeding it the given inputs, then 'exit'."""
    inputs = iter([*user_inputs, "exit"])
    return bot.chat_loop(model, read_input=lambda _prompt: next(inputs))


# ---------- build_model ----------

def test_build_model_uses_openrouter_settings(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    model = bot.build_model()

    assert model.model_name == "google/gemma-4-31b-it:free"
    assert model.openai_api_base == "https://openrouter.ai/api/v1"
    assert model.openai_api_key.get_secret_value() == "test-key"
    assert model.request_timeout == 60
    assert model.max_retries == 1


def test_build_model_has_fallback_models(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    model = bot.build_model()

    assert model.extra_body["models"] == [
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
    ]


# ---------- stream_reply ----------

def test_stream_reply_joins_chunks_and_prints_them(capsys):
    model = FakeModel(chunks("Hello", ", ", "world"))

    reply = bot.stream_reply(model, [HumanMessage(content="hi")])

    assert reply == "Hello, world"
    assert "Hello, world" in capsys.readouterr().out


def test_stream_reply_skips_empty_chunks():
    model = FakeModel(chunks("", "Hi", "", "!"))

    assert bot.stream_reply(model, []) == "Hi!"


def test_stream_reply_with_only_empty_chunks_returns_empty_string():
    model = FakeModel(chunks("", ""))

    assert bot.stream_reply(model, []) == ""


# ---------- chat_loop ----------

def test_exit_ends_chat_immediately():
    model = FakeModel()

    history = run_chat(model)

    assert history == []
    assert model.calls == []


@pytest.mark.parametrize("word", ["exit", "EXIT", "  Exit  "])
def test_exit_is_case_and_whitespace_insensitive(word):
    model = FakeModel()
    inputs = iter([word])

    history = bot.chat_loop(model, read_input=lambda _p: next(inputs))

    assert history == []


def test_history_records_user_and_ai_messages_in_order():
    model = FakeModel(chunks("Hi ", "there"))

    history = run_chat(model, "hello")

    assert [type(m) for m in history] == [HumanMessage, AIMessage]
    assert history[0].content == "hello"
    assert history[1].content == "Hi there"


def test_model_remembers_earlier_messages():
    model = FakeModel(chunks("Nice to meet you, Sam"), chunks("Your name is Sam"))

    run_chat(model, "My name is Sam", "What is my name?")

    # The second call must include the whole first exchange plus the new question.
    second_call = [(type(m), m.content) for m in model.calls[1]]
    assert second_call == [
        (HumanMessage, "My name is Sam"),
        (AIMessage, "Nice to meet you, Sam"),
        (HumanMessage, "What is my name?"),
    ]


def test_error_is_shown_and_user_message_is_removed(capsys):
    model = FakeModel(RuntimeError("Connection error."))

    history = run_chat(model, "hello")

    assert "Error - Connection error." in capsys.readouterr().out
    assert history == []


def test_chat_continues_after_an_error():
    model = FakeModel(RuntimeError("boom"), chunks("Back online"))

    history = run_chat(model, "first try", "second try")

    # Only the successful exchange is kept, and the failed message is not resent.
    assert [m.content for m in history] == ["second try", "Back online"]
    assert [m.content for m in model.calls[1]] == ["second try"]
