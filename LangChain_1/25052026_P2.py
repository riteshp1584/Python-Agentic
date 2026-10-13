import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Dict, Any
from operator import add
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode # <-- NEW: The automated math execution node
from langchain_ollama import ChatOllama

load_dotenv()

# 1. THE STATE DEFINITION

class TeamState (TypedDict):
    messages: Annotated[list[BaseMessage], add]
    next_agent : str

# 2. THE MATHEMATICAL TOOL

# @tool("calculate_sharpe_ratio", description="Calculates the Sharpe Ratio of an investment portfolio to evaluate risk-adjusted performance.")
def calculate_sharpe_ratio(portfolio_return : float, risk_free_rate : float, standard_deviation : float) -> float:
    print(f"\n⚡ [SYSTEM MATH ENGINE] Firing local Python function...")
    return (portfolio_return - risk_free_rate) / standard_deviation

# Explicitly cast the function into a formal LangChain StructuredTool object
from langchain_core.tools import StructuredTool

sharpe_tool_object = StructuredTool.from_function(
    func=calculate_sharpe_ratio,
    name="calculate_sharpe_ratio",
    description="Calculates the Sharpe Ratio of an investment portfolio to evaluate risk-adjusted performance."
)

tools_list = [sharpe_tool_object]
tool_executor_node = ToolNode(tools_list) # <-- Wraps tools into a formal graph execution desk

# 3. INITIALIZE MODELS

supervisor_llm = ChatOllama(
    model="llama3.1:latest",
    temperature=0.0,
    base_url="http://localhost:11434",
    keep_alive='5s'
)

math_llm = ChatOllama(
    model="mistral:latest",
    temperature=0.0,
    base_url="http://localhost:11434",
    keep_alive='5s'
).bind_tools(tools_list)

validator_llm = ChatOllama(
    model="gemma2:2b",
    temperature=0.1,
    base_url="http://localhost:11434",
    keep_alive='5s'
)

# 4. REINFORCED WORKER NODES

def supervisor_node (state: TeamState):
    print("\n👑 [Supervisor (Llama 3.1)] Assessing current state history...")

    prompt = (
        "You are the Portfolio Supervisor. Your team has two desks:\n"
        "1. 'math_worker': Call this desk first to interpret text numbers and invoke the math tool.\n"
        "2. 'validator_worker': Call this desk ONLY AFTER the 'calculate_sharpe_ratio' tool has run and returned a real number.\n"
        "If you see a tool result containing the calculated number in the history, route to 'validator_worker'. If everything is complete, route to 'FINISH'.\n\n"
        "Respond with exactly one word: 'math_worker', 'validator_worker', or 'FINISH'."
    )

    messages = [HumanMessage(content=prompt)] + state["messages"]
    response = supervisor_llm.invoke(messages)
    decision = response.content.strip().lower()

    if "math" in decision: return {"next_agent" : "math_worker"}
    elif "validate" in decision or "validator" in decision: return {"next_agent" : "validator_worker"}
    else: return {"next_agent" : "FINISH"}

def math_worker_node (state: TeamState):
    print("🔬 [Math Worker (Mistral)] Generating precision tool parameters...")
    # Mistral generates the structured tool calling payload here

    worker_prompt = (
        "You are a strict mathematical extractor. Your ONLY job is to call the "
        "calculate_sharpe_ratio tool with the extracted parameters. "
        "CRITICAL: Do not write any conversational summaries, preambles, or analysis in plain text. "
        "Just execute the tool call and stop talking."
    )

    messages = [HumanMessage(content=worker_prompt)] + state["messages"]
    response = math_llm.invoke(messages)
    return {"messages" : [response]}

def validator_worker_node (state: TeamState):
    print("⚖️ [Validator (Gemma 2)] Inspecting math outputs...")
    prompt = (
        "Look at the actual tool return message in the logs. Provide a short"
        "one-sentence financial evaluation explaining whether this specific Sharpe Ratio is healthy."
    )
    messages = state["messages"] + [HumanMessage(content=prompt)]
    response = validator_llm.invoke(messages)
    return {"messages" : [AIMessage(content=f"Validator Final Review: {response.content}")]}

# 5. GRAPH ASSEMBLY WITH DYNAMIC ROUTING PIPES

builder = StateGraph(TeamState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("math_worker", math_worker_node)
builder.add_node("tools", tool_executor_node) # Register the automatic execution desk
builder.add_node("validator_worker", validator_worker_node)

builder.set_entry_point("supervisor")

# Conditional Router for the Supervisor
def supervisor_router (state: TeamState):
    if state["next_agent"] == "math_worker": return "math_worker"
    elif state["next_agent"] == "validator_worker": return "validator_worker"
    else: return "__end__"

builder.add_conditional_edges("supervisor",
                              supervisor_router,
                              {"math_worker" : "math_worker",
                               "validator_worker" : "validator_worker",
                              "__end__" : END
                               })

# Conditional Router for the Math desk: Does Mistral want to call a tool or just talk?
def math_router (state: TeamState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("   >> Caught Tool Call Command! Routing to Tool Execution Engine...")
        return "tools"
    return "supervisor"

builder.add_conditional_edges("math_worker", math_router,
                              {"tools" : "tools",
                               "supervisor" : "supervisor"})

# Leaf edges back to supervisor desk
builder.add_edge("tools", "supervisor")
builder.add_edge("validator_worker", "supervisor")

graph = builder.compile()

graph.get_graph().print_ascii()

# 6. RUN THE SYSTEM

print("\n--- STARTING FINAL QUANT MULTI-AGENT RUN ---")
query = "Can you check my portfolio performance? Return is 14%, volatility is 18%, and the risk-free rate is 3%."

inputs = {"messages" : [HumanMessage(content=query)], "next_agent" : ""}

for output in graph.stream(inputs, stream_mode="updates"):
    for node, value in output.items():
        print(f"--> Done with node '{node}'")
        if "messages" in value and value["messages"]:
            print(f"    Log Content: {value["messages"][-1].content}\n")

