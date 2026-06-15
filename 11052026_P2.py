from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

# 1. The Folder (Same as before)
class GraphState(TypedDict):
    input_text: str
    processed_text: str
    word_count: str  # (ideally should be 'int' but since Mistral might explain the count, so we'll use a string)

# 2. Desk 1: The Human-coded Worker (Simple Node)
def worker_node(state: GraphState):
    print("--- Desk 1: Adding the Signature ---")
    original = state["input_text"]
    return {"processed_text" : f'{original} - Verified by System'}
#     remember that return element should always belong to class elements, that's when the code structure is correct

def llm_agent_node(state: GraphState):
    print("--- Desk 2: LLM is Analyzing ---")

    # Initialize the "Brain"
    llm = ChatOllama(model="mistral:latest", temperature=0)

    # Pull data from the folder
    text_to_analyze = state['processed_text']

    # LLM decides how to answer
    response = llm.invoke(f"Count the words in this text and tell me why: {text_to_analyze}")

    return {"word_count" : response.content}

# 4. The Map
builder = StateGraph(GraphState)

builder.add_node("worker", worker_node)
builder.add_node("llm_agent", llm_agent_node)

builder.set_entry_point("worker")
builder.add_edge("worker", "llm_agent")
builder.add_edge("llm_agent", END)

# 5. Run the Graph
graph = builder.compile()

initial_input = {"input_text": "Quant finance is fascinating"}
final_output = graph.invoke(initial_input)

print("\n----LLM's Analysis---------")
print(final_output["word_count"])
