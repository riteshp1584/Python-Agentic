from typing import TypedDict
from typing import Annotated
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver  # <-- NEW: The Filing Cabinet

# 1. The Folder
class GraphState(TypedDict):
    chat_history : Annotated[list[str], add]

def assistant_node(state: GraphState):
    print("--- Assistant: Processing current conversation ---")
    # We can still read the history to look at it
    current_history = state.get("chat_history", [])
    print(f"    (Assistant reads history size: {len(current_history)})")
    # We return nothing for 'chat_history' because we didn't generate a new reply yet!
    return {}

'''
# 2. The Desk: A Chat Node that appends new thoughts
def assistant_node(state: GraphState):
    print("--- Assistant: Processing current conversation ---")
    history = state.get("chat_history", [])
    return {"chat_history" : history}

'''

# 3. Building the Map
builder = StateGraph(GraphState)
builder.add_node("assistant", assistant_node)
builder.set_entry_point("assistant")
builder.add_edge("assistant", END)

# 4. COMPILING WITH MEMORY
# We initialize our filing cabinet here
memory_cabinet = MemorySaver()

# When we compile, we give the manager access to the cabinet
graph = builder.compile(checkpointer=memory_cabinet)

# 5. RUNNING CONVERSATION ROUND 1
# To use memory, we MUST provide a "thread_id" config passport
# config_1 = {"configurable" : {"thread_id" : "client_project_42"}}
config_1 = {"configurable" : {"thread_id" : "client_project_42"}}

print("--- STARTING ROUND 1 ---")
input_1 = {"chat_history" : ["User : Hello"]}
result_1 = graph.invoke(input_1, config=config_1)
print("Folder state after Round 1:", result_1["chat_history"])

print("\n" + "="*40 + "\n")

# 6. RUNNING CONVERSATION ROUND 2 (The Test)
print("--- STARTING ROUND 2 (Same Thread) ---")
input_2 = {"chat_history" : ["User: Can you help me calculate alpha?"]}
result_2 = graph.invoke(input_2, config=config_1)
print("Folder state after Round 2:", result_2["chat_history"])
