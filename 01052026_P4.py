import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Using dotenv to fetch Google API Key from .env file
load_dotenv()

chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
messages = [SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Give me a joke on California's obsession on technology")]

response = chat.invoke(messages)
print(response.content)
