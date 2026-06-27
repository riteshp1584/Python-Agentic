import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from operator import add
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

# Prebuilt blocks to handle tool execution and routing logic automatically
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# 1. THE TOOL (The Calculator on the Desk)
@tool
def calculate_sharpe_ratio(portfolio_return : float, risk_free_rate : float, standard_deviation : float) -> float:
    """
        Calculates the Sharpe Ratio of an investment portfolio to evaluate risk-adjusted performance.
        Use this tool whenever a user asks to calculate a Sharpe ratio or risk metrics.
    """
    print(f"\n⚡ [SYSTEM TOOL] Executing Python math function...")
    return (portfolio_return - risk_free_rate) / standard_deviation

# 2. THE FOLDER (State) - Exactly like your previous memory script
class GraphState(TypedDict):
    messages : Annotated[list[BaseMessage], add]

# 3. INITIALIZE GEMINI & BIND THE TOOLS
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

# We register our tool list with the model so Gemini knows it has permission to use it
tools_list = [calculate_sharpe_ratio]
llm_with_tools = llm.bind_tools(tools_list)

# 4. THE NODES (The Desks)
def assistant_node(state: GraphState):
    print("--- Assistant: AI is evaluating conversation history... ---")
    chat_history = state["messages"]

    # We call the tool-aware version of our LLM here
    response = llm_with_tools.invoke(chat_history)

    # Return the AI message update
    return {"messages": [response]}

# 5. BUILDING THE LAYOUT (Connecting the Pipes)
builder = StateGraph(GraphState)
# Register our desks
builder.add_node("llm_assistant", assistant_node)
# ToolNode is a prebuilt desk that automatically maps and runs functions in tools_list
builder.add_node("tools_desk", ToolNode(tools_list))
# Set entry point
builder.set_entry_point("llm_assistant")

# THE CONDITIONAL EDGE: The Dynamic Routing Path
builder.add_conditional_edges(
    "llm_assistant", # After the assistant finishes...
    tools_condition, # ...pass it to LangGraph's prebuilt traffic cop function
    {
        "tools" : "tools_desk", # If Gemini requested a tool call -> Route to tools_desk
        "__end__" : END # If Gemini responded with normal text -> Exit to END
    }
)

# THE CYCLE: Loop back to the assistant after the tool prints its answer
builder.add_edge("tools_desk", "llm_assistant")

graph = builder.compile()

# 6. RUN THE TEST
print("\n--- STARTING AGENT RUN ---")
query = "Hey! Can you help me look at my portfolio? My annualized return is 14%, volatility is 18%, and the risk-free rate is sitting at 3%. What is my Sharpe ratio?"
inputs = {"messages" : [('user', query)]}

# Streaming updates lets us see the folder step from node to node in real-time
for output in graph.stream(inputs, stream_mode="updates"):
    for node, value in output.items():
        print(f"--> Finished Node: '{node}'\n")
