# LLM.py - Simple LLM with Ollama (Updated)
from langchain_ollama import OllamaLLM

# Initialize Ollama with Llama 3 (no warnings)
llm = OllamaLLM(model="llama3")

def ask_llm(question):
    response = llm.invoke(question)
    return response

# Test with fraud-related questions
if __name__ == "__main__":
    questions = [
        "Hi, how r u?"
    ]
    
    for question in questions:
        print(f"Question: {question}")
        answer = ask_llm(question)
        print(f"Answer: {answer}\n{'-'*50}\n")