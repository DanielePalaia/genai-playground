"""Main entry point for the GenAI playground."""

from genai_playground.langchain_utils import LangChainHandler


def main() -> None:
    """Run the main application."""
    try:
        # Initialize LangChain handler
        handler = LangChainHandler()

        print("Generating response (this might take a moment)...")
        result: str = handler.generate_response(topic="artificial intelligence")
        print("\nGenerated Response:")
        print(result)

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure Ollama is installed and running.")
        print("Install from: https://ollama.ai/")


if __name__ == "__main__":
    main()
