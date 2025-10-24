from langchain_groq import ChatGroq

# Get free API key from groq AI
llm = ChatGroq(
    api_key="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",
    model="llama-3.3-70b-versatile"  # Free & fast
)

response = llm.invoke("What is fraud detection?")
print(response.content)