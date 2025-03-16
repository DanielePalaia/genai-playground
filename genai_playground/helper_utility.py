# helper_utility.py
from langchain.document_loaders import WebBaseLoader, PyPDFLoader
from typing import List
from langchain.schema import Document


class HelperUtilityClass:
    """Utility class for loading documents using LangChain loaders."""

    @staticmethod
    def load_website(url: str) -> List[Document]:
        """Load documents from a website using WebBaseLoader.

        Args:
            url: The URL of the website to load.

        Returns:
            List of documents.
        """
        print(f"Loading website: {url}...")
        loader = WebBaseLoader(url)
        return loader.load()

    @staticmethod
    def load_pdf(url: str) -> List[Document]:
        """Load documents from a PDF using PyPDFLoader.

        Args:
            url: The URL of the PDF to load.

        Returns:
            List of documents.
        """
        print(f"Loading PDF: {url}...")
        loader = PyPDFLoader(url)
        return loader.load()