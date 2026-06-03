import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda

# 1. Ensure API Key is set
# Using dotenv to fetch Google API Key from .env file
load_dotenv()

def calculate_financials(data):
    """Calculates a basic profit margin from raw input."""
    revenue = data["revenue"]
    costs = data["costs"]
    margin = ((revenue - costs)/revenue) * 100

    return {
        "company": data["company"],
        "margin": f'{margin:.2f}',
        "raw_context": data.get("context", "N/A")
    }

# 2. Setup the Components
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# The prompt now expects the keys returned by our function
prompt = ChatPromptTemplate.from_template(
    "The company {company} has a profit margin of {margin}. "
    "Based on this additional context: {raw_context}, "
    "provide a 1-sentence financial outlook."
)

# 3. Build the Sequence
# We use RunnableLambda to wrap our custom function
chain = RunnableLambda(calculate_financials) | prompt | llm | StrOutputParser()

# 4. Invoke with Raw Data
raw_data = {
    "company": "TechNovus",
    "revenue": 500000,
    "costs": 350000,
    "context": "They are expanding into the APAC market next quarter."
}

result = chain.invoke(raw_data)
print(result)
