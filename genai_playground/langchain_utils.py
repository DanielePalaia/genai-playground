"""LangChain utilities for the GenAI playground."""

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama


class LangChainHandler:
    """Handler for LangChain operations."""

    model_name: str
    llm: Ollama

    def __init__(self, model_name: str = "tinyllama") -> None:
        """Initialize the handler with a specific model.

        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.llm = self._initialize_model()

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

    def create_chain(self) -> LLMChain:
        """Create a simple LangChain chain.

        Returns:
            Configured LLMChain instance
        """
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="Write a short paragraph about {topic}.",
        )

        return LLMChain(llm=self.llm, prompt=prompt)

    def generate_response(self, topic: str) -> str:
        """Generate a response for a given topic.

        Args:
            topic: The topic to generate text about

        Returns:
            Generated text response
        """
        chain = self.create_chain()
        return str(chain.run(topic=topic))
