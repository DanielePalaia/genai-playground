# main.py
from genai_playground.langchain_utils import LangChainHandler
import sys


def main() -> None:
    """Run the main application."""
    # Take the argument
    if len(sys.argv) != 2:
        raise ValueError("You need to specify if you want the chatbot to work with Gemfire, Greenplum, or RabbitMQ.")

    augmented_doc = sys.argv[1]

    if augmented_doc not in ["Gemfire", "Greenplum", "RabbitMQ"]:
        raise ValueError("You can only specify Gemfire, Greenplum, or RabbitMQ.")

    try:
        # Initialize LangChain handler
        handler = LangChainHandler(augmented_doc=augmented_doc)

        print("Chat initialized. Type your messages (Ctrl+C to exit)")
        print("-" * 50)

        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Generate and print response
                print("\nAI: ", end="", flush=True)
                response: str = handler.generate_response(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\n\nGoodbye! Thanks for chatting!")
                break

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure Ollama is installed and running.")
        print("Install from: https://ollama.ai/")


if __name__ == "__main__":
    main()