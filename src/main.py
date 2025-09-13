import requests
import sys
import os
from groq import Groq
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import grok_api_key
# Replace with your actual Grok API key

API_KEY=grok_api_key


client = Groq(
    api_key=grok_api_key,
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
        }
    ],
    model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)