"""LangChain utilities for the GenAI playground."""

from typing import Any

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama


class LangChainHandler:
    """Handler for LangChain operations."""

    model_name: str
    llm: Ollama
    conversation: ConversationChain

    def __init__(self, model_name: str = "orca-mini") -> None:
        """Initialize the handler with a specific model.

        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.llm = self._initialize_model()
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

    def _create_conversation(self) -> ConversationChain:
        """Create a conversation chain with memory.

        Returns:
            Configured ConversationChain instance
        """
        # Create a memory instance to store conversation
        memory = ConversationBufferMemory(ai_prefix="Assistant")

        # Create a conversation prompt with cleaner format
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template="""You are a friendly and capable AI assistant chatting with a human.
            When they say "Hi" or "Hello", greet them back warmly.
            When they ask questions, answer them directly.
            Keep your responses natural and friendly.

            Previous messages:
            {history}

            Human: {input}
            Assistant:""",
        )

        # Create and return the conversation chain
        return ConversationChain(
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
        return str(self.conversation.predict(input=user_input))
