"""Main entry point for the GenAI playground."""

from genai_playground.langchain_utils import LangChainHandler
import requests
from io import BytesIO
import PyPDF2
    
def extract_text_from_pdf(url: str) -> str:
    # Download the PDF file
    response = requests.get(url)
    pdf_file = BytesIO(response.content)
    
    # Read the PDF file
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    
    # Extract text from each page
    for page in reader.pages:
        text += page.extract_text()
    
    return text

def main() -> None:
    """Run the main application."""
    try:
        
        additional_info = extract_text_from_pdf("https://techdocs.broadcom.com/content/dam/broadcom/techdocs/us/en/pdf/vmware-tanzu/data-solutions/tanzu-gemfire/10-1/gf/gf.pdf")
        # Initialize LangChain handler
        handler = LangChainHandler(additional_info=additional_info)

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
                response: str = handler.generate_response(user_input, additional_info=additional_info)
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
