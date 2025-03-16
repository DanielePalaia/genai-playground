# tests/test_similarity_db.py
import pytest
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings

from genai_playground.similarity_database import SimilarityDatabaseManagement


@pytest.fixture
def embeddings():
    return OllamaEmbeddings(model="orca-mini")


@pytest.fixture
def documents():
    return [
        Document(page_content="This is a test document.", metadata={"source": "test"}),
        Document(page_content="Another test document.", metadata={"source": "test"}),
    ]


def test_create_vector_db(embeddings, documents):
    """Test creating a vector database."""
    db_manager = SimilarityDatabaseManagement(embeddings, persist_directory="./test_db")
    vector_db = db_manager.create_vector_db(documents)
    assert vector_db is not None, "Vector database creation failed."


def test_similarity_search(embeddings, documents):
    """Test similarity search in the vector database."""
    db_manager = SimilarityDatabaseManagement(embeddings, persist_directory="./test_db")
    db_manager.create_vector_db(documents)
    results = db_manager.similarity_search("test", k=2)
    assert len(results) == 2, "Similarity search returned incorrect number of results."
