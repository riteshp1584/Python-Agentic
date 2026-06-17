# from IPython.terminal.shortcuts.filters import pass_through
from typing import TypedDict
from langgraph.graph import StateGraph, END

# 1. The Folder
class GraphState(TypedDict):
    input_text: str
    iteration_count: int  # We'll track how many times it loops

# 2. Desk 1: The Worker
def worker_node(state: GraphState):
    count = state.get("iteration_count", 0) + 1
    # We use .get() to handle the first run where the count doesn't exist yet
    print(f"--- Worker: Attempt #{count} ---")

    # We simulate "improving" the text by adding more words each time

    new_text = state["input_text"] + " word"
    return {"input_text" : new_text, "iteration_count" : count}

# 3. The Router: Quality Control (The Translator)
def checker_logic(state: GraphState):
    print("--- Checker: Verifying length ---")
    # If text is less than 30 characters, it's "low_quality"
    if len(state["input_text"]) < 30:
        return "redo"
    else:
        return "pass"

# 4. Building the Map
builder = StateGraph(GraphState)

# Register the desk
builder.add_node("worker_desk", worker_node)

# Start here
builder.set_entry_point("worker_desk")

# THE TRANSLATOR BRIDGE (The Cycle)
builder.add_conditional_edges(
    "worker_desk",    # After this desk...
    checker_logic,    # ...ask the checker...
    {
        "redo": "worker_desk", # "If checker says 'redo', go BACK to worker_desk"
        "pass": END            # "If checker says 'pass', go to the EXIT"
    }
)


# 5. Execute
graph = builder.compile()

print("Starting the process with 'Start'...")
final_result = graph.invoke({"input_text": "Start", "iteration_count": 0})

print("\n--- FINAL OUTPUT ---")
print(f"Text: {final_result['input_text']}")
print(f"Total Attempts: {final_result['iteration_count']}")
