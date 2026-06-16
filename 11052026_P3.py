from typing import TypedDict
from langgraph.graph import StateGraph, END

# 1. The Folder
class GraphState(TypedDict):
    input_text : str
    category : str

# 2. The Desks (Nodes)
def finance_node(state: GraphState):
    print("--- Routing to: FINANCE DESK ---")
    return {"category" : "Financial Services"}

def general_node(state: GraphState):
    print("--- Routing to: GENERAL DESK ---")
    return {"category" : "General Inquiry"}

# 3. The Router Logic (This is NOT a node, it's a guide)
def route_decision(state: GraphState):
    # This function acts like a traffic cop
    if "money" in state["input_text"]:
        return "finance_dept"
    else:
        return "general_dept"

# 4. Building the Map
builder = StateGraph(GraphState)

# Add our two destination desks
builder.add_node("finance", finance_node)
builder.add_node("general", general_node)

# --- THE CONDITIONAL PATHWAY ---
# Instead of a fixed arrow, we add a logic gate at the start

builder.set_conditional_entry_point(route_decision,
                                    {
                                        "finance_dept" : "finance",
                                        "general_dept" : "general"
                                    })

# Both desks lead to the exit
builder.add_edge("finance", END)
builder.add_edge("general", END)

# 5. Run it twice to see the difference
graph = builder.compile()

print("Test 1: 'I want to save money'")
graph.invoke({"input_text" : "I want to save money"})

print("\n Test 2: 'How is the weather?")
graph.invoke({'input_text' : 'How is the weather?'})
