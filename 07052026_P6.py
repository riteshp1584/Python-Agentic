from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize the Model (The 'L' in LCEL)
llm = ChatOllama(model="mistral:latest", temperature=0, keep_alive='5s')

# 2. Create the Template (The 'P' in LCEL)
prompt = ChatPromptTemplate.from_template(
    "As a Quant Dev, write a Python function to calculate {indicator} for a given series."
)

# 3. Combine into a Chain
# Input -> Prompt -> LLM -> String Output
chain = prompt | llm | StrOutputParser()

# 4. Use the Chain
# This is much cleaner than manually formatting strings
result = chain.invoke({"indicator": "Relative Strength Index (RSI)"})
print(result)
