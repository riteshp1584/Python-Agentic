import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough

# Ensure API Key is set
# Using dotenv to fetch Google API Key from .env file
load_dotenv()

# Initialize the model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# First chain generates a story
story_prompt = ChatPromptTemplate.from_template("Write a short story about {topic}")
story_chain = story_prompt | llm | StrOutputParser()

# Second chain analyzes the story
analysis_prompt = ChatPromptTemplate.from_template('Analyse the mood of the following: \n{story}')
analysis_chain = analysis_prompt | llm | StrOutputParser()

enhanced_chain = RunnablePassthrough.assign(story=story_chain).assign(analysis=analysis_chain)

result = enhanced_chain.invoke({"topic" : "a day in Amazon rainforest"})

print(result["story"])
