import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Ensure API Key is set
# Using dotenv to fetch Google API Key from .env file
load_dotenv()

# 2. Initialize Model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# 3. Define Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

# 4. Create Chain
chain = prompt | llm | StrOutputParser()

# 5. Invoke with Dictionary
try:
    response = chain.invoke({"input": "Who are the most profound thinkers of stoic philosophy?"})
    print("--- RESPONSE ---")
    print(response)
except Exception as e:
    print(f"An error occurred: {e}")
