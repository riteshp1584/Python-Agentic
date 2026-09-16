from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize Mistral
# We set temperature to 0 for consistent, "non-creative" code
llm = ChatOllama(model="mistral", temperature=0)

# 2. Concept: ChatPromptTemplate
# This separates the 'Instructions' from the 'Subject'
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Quant Developer. Provide only the Python function requested. No conversational filler."),
    ("human", "Write a function to calculate the {factor_name} using a {window} period.")
])

# 3. Concept: The Pipe Operator (|)
# This is the 'Chain' in LangChain.
# Input Dictionary -> Prompt -> LLM -> String Output
chain = prompt | llm | StrOutputParser()

# 4. Invoke the chain
# This makes your code modular and reusable
result = chain.invoke({
    "factor_name": "Exponential Moving Average (EMA)",
    "window": "20"
})

print("--- CLEAN OUTPUT ---")
print(result)
