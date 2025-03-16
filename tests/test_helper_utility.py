# tests/test_helper_utility.py
import pytest

from genai_playground.helper_utility import HelperUtilityClass


def test_load_website():
    """Test loading a website."""
    url = "https://www.rabbitmq.com/"
    documents = HelperUtilityClass.load_website(url)
    assert len(documents) > 0, "No documents were loaded from the website."


def test_load_pdf():
    """Test loading a PDF."""
    url = "https://techdocs.broadcom.com/content/dam/broadcom/techdocs/us/en/pdf/vmware-tanzu/data-solutions/tanzu-gemfire/10-1/gf/gf.pdf"
    documents = HelperUtilityClass.load_pdf(url)
    assert len(documents) > 0, "No documents were loaded from the PDF."
