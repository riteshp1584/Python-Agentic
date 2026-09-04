from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

local_llm = ChatOllama(
    model="llama3.1",
    temperature=0.7,
    keep_alive=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a local AI running on Ritesh's machine."),
    ("human", "{input}")
])

chain = prompt | local_llm | StrOutputParser()

try:
    result = chain.invoke({"input" : "what are the advantages of precious metals as assets in capital markets?"})
    print(result)
except Exception as e:
    print(f"Make sure the Ollama app is running! Error: {e}")
