from groq import Groq

client = Groq(api_key="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")
models = client.models.list()

for model in models.data:
    print(model.id)

# I picked the "llama-3.3-70b-versatile" model.