from typing import TypedDict
from langgraph.graph import StateGraph, END

# 1. Define the 'State'
class GraphState(TypedDict):
    input_text : str
    processed_text : str

# 2. Define a Node (The Worker)
def worker_node(state: GraphState):
    print("--- Node: Processing Text ---")
    original = state["input_text"]
    return {"processed_text": f"{original}... processed by the graph!"}

# 3. Construct the Graph
builder = StateGraph(GraphState)
builder.add_node("my_worker", worker_node)

# Define the flow: Entry -> Node -> Finish
builder.set_entry_point("my_worker")
builder.add_edge("my_worker", END)

graph = builder.compile()

initial_state = {"input_text" : "Hello, LangGraph"}
final_output = graph.invoke(initial_state)
