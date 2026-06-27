from langchain_classic.chains import llm
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

local_llm = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0.7,
    keep_alive=0
)

prompt = ChatPromptTemplate.from_template("Explain {concept} in simple terms")

chain = prompt | local_llm | StrOutputParser()

result = chain.invoke({"concept" : "machine learning"})

print(result)
