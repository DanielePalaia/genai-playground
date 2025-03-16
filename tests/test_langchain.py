# tests/test_langchain.py
import pytest

from genai_playground.langchain_utils import LangChainHandler


def test_langchain_handler_initialization():
    """Test initialization of LangChainHandler."""
    handler = LangChainHandler(augmented_doc="Gemfire")
    assert handler is not None, "LangChainHandler initialization failed."


def test_generate_response():
    """Test generating a response from the LangChainHandler."""
    handler = LangChainHandler(augmented_doc="Gemfire")
    response = handler.generate_response("What is Gemfire?")
    assert isinstance(response, str), "Response should be a string."
    assert len(response) > 0, "Response should not be empty."
