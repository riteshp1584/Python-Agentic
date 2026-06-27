
from langchain_community.llms import FakeListLLM

fake_llm = FakeListLLM(responses=["Always this response!"])

result = fake_llm.invoke("What is the best way to tour Europe?")

print(result)
