"""LangChain utilities for the GenAI playground."""

from typing import Any

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from io import BytesIO

class LangChainHandler:
    """Handler for LangChain operations."""

    model_name: str
    llm: Ollama
    conversation: LLMChain

    def __init__(self, model_name: str = "orca-mini", additional_info:str="") -> None:
        """Initialize the handler with a specific model.

        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.llm = self._initialize_model()
        self.conversation = self._create_conversation(additional_info)

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

    def _create_conversation(self, additional_info: str) -> LLMChain:
        """Create a conversation chain with memory.

        Returns:
            Configured ConversationChain instance
        """
        # Create a memory instance to store conversation
        memory = ConversationBufferMemory(memory_key="history", input_key="input")

    
        # Create a conversation prompt with cleaner format
        prompt = PromptTemplate(
            input_variables=["input", "history", "additional_info"],  # Ensure these match what you pass
            template="Giving priority to the following information: {additional_info}\n\nConversation History:\n{history}\n\nAnswer the question: {input}"
            
        )

        # Create and return the conversation chain
        return LLMChain(
            llm=self.llm,
            memory=memory,
            prompt=prompt,
            verbose=False,
        )

    def generate_response(self, user_input: str, additional_info: str) -> str:
        """Process user input and return AI response.

        Args:
            user_input: The user's input message

        Returns:
            AI's response
        """
        return str(self.conversation.predict(input=user_input, additional_info=additional_info))
