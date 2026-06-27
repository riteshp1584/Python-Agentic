from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


'''
template = ChatPromptTemplate([
    ("system", "you are a problem solving assistant"),
    ("user", "{problem}")
])

chat = ChatOllama(model="gemma2:2b",
                  temperature=0.3,
                  keep_alive='5s')

chain = chat | template

response = chain.invoke({"problem" : ""})
'''


# 1. Initialize with the reasoning toggle
llm = ChatOllama(
    model="gemma2:2b",
    temperature=0.4
    # This captures reasoning separately in additional_kwargs['reasoning_content']
)

# 2. Invoke
response = llm.invoke("Calculate the impact of a 50bps rate hike on a 10-year bond price.")

# 3. Accessing the 'Thought' vs 'Answer'
# If reasoning=True, the "thinking" is stripped from the main content
print("--- THE THINKING ---")
print(response.additional_kwargs.get("reasoning_content"))

print("--- THE FINAL ANSWER ---")
print(response.content)
