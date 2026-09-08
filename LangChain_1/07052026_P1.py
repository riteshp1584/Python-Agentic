from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

chat = ChatOllama(model="gemma2:2b",
                  temperature=0.3,
                  keep_alive='5s')

messages = [
    SystemMessage(content="You are a AI assistant with good knowledge of Python programming language."),
    HumanMessage(content="Which are the most important libraries for Machine Learning applications?")
]

response = chat.invoke(messages)
print(response.content)
