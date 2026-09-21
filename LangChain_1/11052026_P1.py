from typing import TypedDict
from langgraph.graph import StateGraph, END

# 1. Our Folder now needs an extra slot for the count
class GraphState(TypedDict):
    input_text: str
    processed_text: str
    word_count: int

# 2. Worker Node (Desk 1)
def worker_node(state: GraphState):
    print("--- Desk 1: The Worker is signing the paper ---")
    original = state["input_text"]
    return {"processed_text": f"{original} - Verified by Agent"}

# 3. Analyzer Node (Desk 2)
def analyzer_node(state: GraphState):
    print("--- Desk 2: The Analyzer is counting words ---")
    # This desk looks at the work done by Desk 1
    text_to_measure = state["processed_text"]
    count = len(text_to_measure.split())
    return {"word_count": count}

# 4. Building the Floor Plan
builder = StateGraph(GraphState)

builder.add_node("worker", worker_node)
builder.add_node("analyzer", analyzer_node)

# --- THE PATHWAY ---
# Start -> Worker -> Analyzer -> Finish
builder.set_entry_point("worker")
builder.add_edge("worker", "analyzer") # The "Hand-off"
builder.add_edge("analyzer", END)      # The "Exit"

# 5. Compile and Run
graph = builder.compile()

# ASCII diagram to show flows
graph.get_graph().print_ascii()

# Drop off the folder
initial_state = {"input_text": "LangGraph is great"}
final_output = graph.invoke(initial_state)

print("\n--- FINAL FOLDER CONTENT ---")
print(final_output)
