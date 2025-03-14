# helper_utility.py
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF


class HelperUtilityClass:
    """Utility class for HTTP requests and PDF parsing."""

    @staticmethod
    def scrape_website(url: str) -> str:
        """Scrape text content from a website.

        Args:
            url: The URL of the website to scrape.

        Returns:
            Extracted text content.
        """
        print(f"Scraping {url}...")
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text(separator=" ")

    @staticmethod
    def extract_text_from_pdf(url: str, max_pages: int = 10) -> str:
        """Extract text content from a PDF using PyMuPDF.

        Args:
            url: The URL of the PDF to extract.
            max_pages: Maximum number of pages to extract.

        Returns:
            Extracted text content.
        """
        print(f"Extracting text from {url} (first {max_pages} pages)...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            pdf_file = BytesIO(response.content)
            text = ""
            with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
                for i, page in enumerate(doc):
                    if i >= max_pages:
                        break
                    text += page.get_text()
            print(f"Extracted Text Length: {len(text)}")
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""