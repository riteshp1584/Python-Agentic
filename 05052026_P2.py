from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(
    model="gemma2:2b",
    temperature=0.3,
    num_ctx=4096,
    keep_alive='5s'
)

prompt = ChatPromptTemplate.from_template("Describe the features of {topic}.")

chain = prompt | llm

response = chain.invoke({"topic": "Indian Constitution"})

print(response.content)
