import os
from io import BytesIO
from typing import Any, List

import fitz  # PyMuPDF
import PyPDF2
import requests
from bs4 import BeautifulSoup
from langchain.chains import LLMChain
from langchain.embeddings import OllamaEmbeddings  # Use Ollama embeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import (
    CharacterTextSplitter,  # For splitting text into chunks
)
from langchain.vectorstores import Chroma
from langchain_community.llms import Ollama


class LangChainHandler:
    """Handler for LangChain operations."""

    def __init__(self, model_name: str = "orca-mini", augemented_doc = "Gemfire") -> None:
        """Initialize the handler with a specific model.

        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.llm = self._initialize_model()
        self.embeddings = OllamaEmbeddings(model=model_name)  # Initialize embeddings
        self._augemented_doc = augemented_doc
        self.vector_db = (
            self._load_or_create_vector_db()
        )  # Load or create vector database
        self.conversation = self._create_conversation()

    def _initialize_model(self) -> Ollama:
        """Initialize a local Ollama model.

        Returns:
            Configured Ollama model instance
        """
        print(f"Connecting to Ollama with model: {self.model_name}")
        return Ollama(
            model=self.model_name,
            temperature=0.7,
        )

    def _scrape_website(self, url: str) -> str:
        """Scrape text content from a website.

        Args:
            url: The URL of the website to scrape

        Returns:
            Extracted text content
        """
        print(f"Scraping {url}...")
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text(separator=" ")

    def _extract_text_from_pdf(self, url: str, max_pages: int = 10) -> str:
        """Extract text content from a PDF using PyMuPDF."""
        print(f"Extracting text from {url} (first {max_pages} pages)...")
        try:
            # Download the PDF
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad status codes
            pdf_file = BytesIO(response.content)

            # Extract text from the PDF
            text = ""
            with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
                for i, page in enumerate(doc):
                    if i >= max_pages:  # Limit the number of pages processed
                        break
                    text += page.get_text()

            print(f"Extracted Text Length: {len(text)}")
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""  # Return empty string if extraction fails

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text to ensure it has proper breaks for splitting.

        Args:
            text: The text to preprocess

        Returns:
            Preprocessed text with proper breaks
        """
        print("Preprocessing text...")
        # Add newlines after periods, question marks, and exclamation marks
        text = text.replace(". ", ".\n").replace("? ", "?\n").replace("! ", "!\n")

        # Add newlines after common sentence-ending patterns
        text = text.replace("; ", ";\n").replace(": ", ":\n")

        # Remove excessive whitespace
        text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

        return text

    def _load_or_create_vector_db(self) -> Chroma:
        """Load or create a vector database with updated information from the web.

        Returns:
            Configured Chroma vector store
        """
        persist_directory = "./chroma_db_" + self._augemented_doc
        if os.path.exists(persist_directory):
            print("Loading existing vector database...")
            return Chroma(
                persist_directory=persist_directory, embedding_function=self.embeddings
            )
        else:
            print("Creating new vector database...")
            return self._create_vector_db()

    def _create_vector_db(self) -> Chroma:
        """Create a vector database with updated information from the web.

        Returns:
            Configured Chroma vector store
        """

        if self._augemented_doc == "RabbitMQ":
            # Scrape RabbitMQ website
            rabbitmq_text = self._scrape_website("https://www.rabbitmq.com/")
            rabbitmq_text = self._preprocess_text(rabbitmq_text)

        if self._augemented_doc == "Gemfire":
            # Extract text from Gemfire PDF
            gemfire_text = self._extract_text_from_pdf(
                "https://techdocs.broadcom.com/content/dam/broadcom/techdocs/us/en/pdf/vmware-tanzu/data-solutions/tanzu-gemfire/10-1/gf/gf.pdf"
            )
            gemfire_text = self._preprocess_text(gemfire_text)

        if self._augemented_doc == "Greenplum":
            # Extract text from Greenplum PDF
            greenplum_text = self._extract_text_from_pdf(
                "https://techdocs.broadcom.com/content/dam/broadcom/techdocs/us/en/pdf/vmware-tanzu/data-solutions/tanzu-greenplum/6/greenplum-database/greenplum-database.pdf"
            )
            greenplum_text = self._preprocess_text(greenplum_text)


        # Split text into smaller chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=500,  # Split text into chunks of 1000 characters
            chunk_overlap=100,  # Add overlap to maintain context
            separator="\n",  # Split on newlines for better structure
            length_function=len,  # Use Python's built-in len function
        )

        # Create documents from the extracted text
        if self._augemented_doc == "Rabbitmq":
            rabbitmq_docs = text_splitter.split_text(rabbitmq_text)
            print(f"RabbitMQ Chunks: {len(rabbitmq_docs)}")
            # Add metadata to each chunk
            rabbitmq_docs = [
                Document(page_content=chunk, metadata={"source": "RabbitMQ Website"})
                for chunk in rabbitmq_docs
            ]
            documents = rabbitmq_docs
        if self._augemented_doc == "Gemfire":
            gemfire_docs = text_splitter.split_text(gemfire_text)
            print(f"Gemfire Chunks: {len(gemfire_docs)}")
            # Add metadata to each chunk
            gemfire_docs = [
                Document(page_content=chunk, metadata={"source": "Gemfire PDF"})
                for chunk in gemfire_docs
            ]
            documents = gemfire_docs
        if self._augemented_doc == "Greenplum":
            greenplum_docs = text_splitter.split_text(greenplum_text)
            print(f"Greenplum Chunks: {len(greenplum_docs)}")
            # Add metadata to each chunk
            greenplum_docs = [
                Document(page_content=chunk, metadata={"source": "Greenplum PDF"})
                for chunk in greenplum_docs
            ]
            documents = greenplum_docs

        # Create the vector database
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db_" + self._augemented_doc,  # Save the database locally
        )

    def _create_conversation(self) -> LLMChain:
        """Create a conversation chain with memory.

        Returns:
            Configured LLMChain instance
        """
        # Create a memory instance to store conversation history
        memory = ConversationBufferMemory(memory_key="history", input_key="input")

        # Create a conversation prompt with a cleaner format
        prompt = PromptTemplate(
            input_variables=[
                "input",
                "history",
                "context",
            ],  # Added "context" for retrieved info
            template="Based on this context: {context}\n\nConversation History:\n{history}\n\nAnswer the question: {input}",
        )

        # Create and return the conversation chain
        return LLMChain(
            llm=self.llm,
            memory=memory,
            prompt=prompt,
            verbose=False,
        )

    def generate_response(self, user_input: str) -> str:
        """Process user input and return AI response.

        Args:
            user_input: The user's input message

        Returns:
            AI's response
        """
        # Retrieve relevant information from the vector database
        retrieved_docs = self.vector_db.similarity_search(
            user_input, k=5
        )  # Retrieve top 2 relevant documents

        context = "\n".join(
            [doc.page_content for doc in retrieved_docs]
        )  # Combine into a single context string

        # Debug: Print retrieved documents
        print("Retrieved Documents:")
        for doc in retrieved_docs:
            print(
                f"Source: {doc.metadata['source']}\nContent: {doc.page_content[:200]}...\n"
            )

        # Pass the context to the conversation chain
        return str(self.conversation.predict(input=user_input, context=context))
