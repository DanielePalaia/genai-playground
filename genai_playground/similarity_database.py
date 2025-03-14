# similarity_database.py
import os
from typing import List

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings


class SimilarityDatabaseManagement:
    """Handles all interactions with the Chroma vector database."""

    def __init__(self, embeddings, persist_directory: str):
        """Initialize the similarity database manager.

        Args:
            embeddings: Embedding model to use for vectorization.
            persist_directory: Directory to persist the vector database.
        """
        self.embeddings = embeddings
        self.persist_directory = persist_directory

    def create_vector_db(self, documents: List[Document]) -> Chroma:
        """Create a new vector database from a list of documents.

        Args:
            documents: List of documents to add to the database.

        Returns:
            Configured Chroma vector store.
        """
        print("Creating new vector database...")
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def load_vector_db(self) -> Chroma:
        """Load an existing vector database from disk.

        Returns:
            Configured Chroma vector store.
        """
        print("Loading existing vector database...")
        return Chroma(
            persist_directory=self.persist_directory, embedding_function=self.embeddings
        )

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform a similarity search on the vector database.

        Args:
            query: The query string.
            k: Number of results to return.

        Returns:
            List of documents most similar to the query.
        """
        vector_db = self.load_vector_db()
        return vector_db.similarity_search(query, k=k)