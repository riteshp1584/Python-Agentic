import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Ensure API Key is set
load_dotenv()
# Using dotenv to fetch Google API Key from .env file

# 2. Initialize Model (Try without the 'models/' prefix first)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# 3. Define Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

# 4. Create Chain
chain = prompt | llm

# 5. Invoke with Dictionary
try:
    response = chain.invoke({"input": "Suggest a tour plan for a family vacation to China?"})
    print("--- RESPONSE ---")
    print(response.content)
except Exception as e:
    print(f"An error occurred: {e}")
