# langchain_utils.py
from typing import Any, List
import os

from langchain.chains import LLMChain
from langchain_community.embeddings import OllamaEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma

from genai_playground.helper_utility import HelperUtilityClass
from genai_playground.similarity_database import SimilarityDatabaseManagement


class LangChainHandler:
    """Handler for LangChain operations."""

    def __init__(self, model_name: str = "orca-mini", augmented_doc: str = "Gemfire") -> None:
        """Initialize the handler with a specific model.

        Args:
            model_name: Name of the Ollama model to use.
            augmented_doc: The document type to augment the model with.
        """
        self.model_name = model_name
        self.augmented_doc = augmented_doc
        self.llm = self._initialize_model()
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.vector_db_manager = SimilarityDatabaseManagement(
            embeddings=self.embeddings,
            persist_directory=f"./chroma_db_{self.augmented_doc}",
        )
        self.vector_db = self._load_or_create_vector_db()
        self.conversation = self._create_conversation()

    def _initialize_model(self) -> Ollama:
        """Initialize a local Ollama model.

        Returns:
            Configured Ollama model instance.
        """
        print(f"Connecting to Ollama with model: {self.model_name}")
        return Ollama(
            model=self.model_name,
            temperature=0.7,
        )

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text to ensure it has proper breaks for splitting.

        Args:
            text: The text to preprocess.

        Returns:
            Preprocessed text with proper breaks.
        """
        print("Preprocessing text...")
        text = text.replace(". ", ".\n").replace("? ", "?\n").replace("! ", "!\n")
        text = text.replace("; ", ";\n").replace(": ", ":\n")
        text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        return text

    def _load_or_create_vector_db(self) -> Chroma:
        """Load or create a vector database with updated information from the web.

        Returns:
            Configured Chroma vector store.
        """
        # Define document sources
        document_sources = {
            "RabbitMQ": ("https://www.rabbitmq.com/", "website"),
            "Gemfire": (
                "https://techdocs.broadcom.com/content/dam/broadcom/techdocs/us/en/pdf/vmware-tanzu/data-solutions/tanzu-gemfire/10-1/gf/gf.pdf",
                "pdf",
            ),
            "Greenplum": (
                "https://techdocs.broadcom.com/content/dam/broadcom/techdocs/us/en/pdf/vmware-tanzu/data-solutions/tanzu-greenplum/6/greenplum-database/greenplum-database.pdf",
                "pdf",
            ),
        }

        if self.augmented_doc not in document_sources:
            raise ValueError(f"Unsupported document: {self.augmented_doc}")

        url, source_type = document_sources[self.augmented_doc]

        # Load documents using LangChain loaders
        print(f"Loading {self.augmented_doc} document...")
        if source_type == "website":
            documents = HelperUtilityClass.load_website(url)  # Use WebBaseLoader
        elif source_type == "pdf":
            documents = HelperUtilityClass.load_pdf(url)  # Use PyPDFLoader

        # Preprocess text
        text = "\n".join([doc.page_content for doc in documents])
        text = self._preprocess_text(text)

        # Split text into smaller chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separator="\n",
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        print(f"{self.augmented_doc} Chunks: {len(chunks)}")

        # Add metadata to each chunk
        documents = [
            Document(page_content=chunk, metadata={"source": self.augmented_doc})
            for chunk in chunks
        ]

        # Create or load the vector database
        if os.path.exists(self.vector_db_manager.persist_directory):
            return self.vector_db_manager.load_vector_db()
        else:
            return self.vector_db_manager.create_vector_db(documents)

    def _create_conversation(self) -> LLMChain:
        """Create a conversation chain with memory.

        Returns:
            Configured LLMChain instance.
        """
        memory = ConversationBufferMemory(memory_key="history", input_key="input")
        prompt = PromptTemplate(
            input_variables=["input", "history", "context"],
            template="Based on this context: {context}\n\nConversation History:\n{history}\n\nAnswer the question: {input}",
        )
        return LLMChain(
            llm=self.llm,
            memory=memory,
            prompt=prompt,
            verbose=False,
        )

    def generate_response(self, user_input: str) -> str:
        """Process user input and return AI response.

        Args:
            user_input: The user's input message.

        Returns:
            AI's response.
        """
        # Retrieve relevant information from the vector database
        retrieved_docs = self.vector_db_manager.similarity_search(user_input, k=5)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Debug: Print retrieved documents
        print("Retrieved Documents:")
        for doc in retrieved_docs:
            print(f"Source: {doc.metadata['source']}\nContent: {doc.page_content[:200]}...\n")

        # Pass the context to the conversation chain
        return str(self.conversation.predict(input=user_input, context=context))