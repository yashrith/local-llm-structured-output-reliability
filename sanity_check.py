import ollama

model = ['llama3.2','llama3.1:8b','qwen2.5:14b']
try:
    print("Testing connection to Ollama...")
    response = ollama.chat(
        model=model[2],
        messages=[{'role': 'user', 'content': 'Say the word "Ready" and nothing else.'}]
    )
    print(f"Ollama response: {response['message']['content'].strip()}")
    print("✅ System check passed! Environment is ready for testing.")
except Exception as e:
    print(f"❌ Connection failed: {e}")