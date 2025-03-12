from typing import Any, List
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.embeddings import OllamaEmbeddings  # Use Ollama embeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import requests
from bs4 import BeautifulSoup
from io import BytesIO
import PyPDF2

class LangChainHandler:
    """Handler for LangChain operations."""

    def __init__(self, model_name: str = "orca-mini") -> None:
        """Initialize the handler with a specific model.

        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.llm = self._initialize_model()
        self.embeddings = OllamaEmbeddings(model=model_name)  # Initialize embeddings
        self.vector_db = self._create_vector_db()  # Create vector database with updated info
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
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text(separator=" ")

    def _extract_text_from_pdf(self, url: str) -> str:
        """Extract text content from a PDF.

        Args:
            url: The URL of the PDF to extract

        Returns:
            Extracted text content
        """
        response = requests.get(url)
        pdf_file = BytesIO(response.content)
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def _create_vector_db(self) -> Chroma:
        """Create a vector database with updated information from the web.

        Returns:
            Configured Chroma vector store
        """
        # Scrape RabbitMQ website
        rabbitmq_text = self._scrape_website("https://www.rabbitmq.com/")

        # Extract text from Gemfire PDF
        gemfire_text = self._extract_text_from_pdf(
            "https://techdocs.broadcom.com/content/dam/broadcom/techdocs/us/en/pdf/vmware-tanzu/data-solutions/tanzu-gemfire/10-1/gf/gf.pdf"
        )

        # Extract text from Greenplum PDF
        greenplum_text = self._extract_text_from_pdf(
            "https://techdocs.broadcom.com/content/dam/broadcom/techdocs/us/en/pdf/vmware-tanzu/data-solutions/tanzu-greenplum/6/greenplum-database/greenplum-database.pdf"
        )

        # Create documents from the extracted text
        documents = [
            Document(page_content=rabbitmq_text, metadata={"source": "RabbitMQ Website"}),
            Document(page_content=gemfire_text, metadata={"source": "Gemfire PDF"}),
            Document(page_content=greenplum_text, metadata={"source": "Greenplum PDF"}),
        ]

        # Create the vector database
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db"  # Save the database locally
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
            input_variables=["input", "history", "context"],  # Added "context" for retrieved info
            template="Given priority to the following context: {context}\n\nConversation History:\n{history}\n\nAnswer the question: {input}"
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
        retrieved_docs = self.vector_db.similarity_search(user_input, k=2)  # Retrieve top 2 relevant documents
        print("retrieved_docs" + str(retrieved_docs))
        context = "\n".join([doc.page_content for doc in retrieved_docs])  # Combine into a single context string

        # Pass the context to the conversation chain
        return str(self.conversation.predict(input=user_input, context=context))