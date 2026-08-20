import os
from dotenv import load_dotenv
from langchain_classic.chains.question_answering.map_reduce_prompt import messages
from typing import TypedDict, Annotated, Dict, Any, Literal
from operator import add
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

load_dotenv()

# 1. THE SHARED TEAM STATE FOLDER

class TeamState(TypedDict):
    messages : Annotated[list[BaseMessage], add]
    next_agent : str # The Supervisor updates this to route the folder

# 2. THE MATHEMATICAL TOOL

# @tool
@tool("calculate_sharpe_ratio", description="Calculates the Sharpe Ratio of an investment portfolio to evaluate risk-adjusted performance.")
def calculate_sharpe_ratio(portfolio_return : float, risk_free_rate : float, standard_deviation : float) -> float:
    # """Calculates the Sharpe Ratio of an investment portfolio."""
    print(f"\n ⚡ [MISTRAL MATH WORKER] Executing Python tool calculation...")
    return (portfolio_return - risk_free_rate) / standard_deviation

# 3. INITIALIZE THE SEPARATE BRAINS

# Each model gets pinned to its specific specialized role
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
).bind_tools([calculate_sharpe_ratio])

validator_llm = ChatOllama(
    model="gemma2:2b",
    temperature=0.1,
    base_url="http://localhost:11434",
    keep_alive='5s'
)

# 4. ENHANCED WORKERS WITH ROLE REINFORCEMENT (NODES)


def supervisor_node(state: TeamState):
    print("\n👑 [Supervisor Desk (Llama 3.1)] Evaluating next steps...")

    prompt = (
        "You are the Portfolio Supervisor Manager. Your team has two workers:\n"
        "1. 'math_worker': Must be called first to extract and compute the Sharpe Ratio.\n"
        "2. 'validator_worker': Must be called after the math is computed to validate the results.\n\n"
        "CRITICAL RULE: If the math has just been processed, you MUST route to 'validator_worker' next. "
        "Only route to 'FINISH' after BOTH workers have completed their tasks.\n\n"
        "Respond with ONLY ONE word matching who should work next: 'math_worker', 'validator_worker', or 'FINISH'."
    )

    messages = [HumanMessage(content=prompt)] + state["messages"]
    response = supervisor_llm.invoke(messages)
    decision = response.content.strip().lower()

    if "math" in decision:
        next_step = "math_worker"
    elif "validate" in decision or "validator" in decision:
        next_step = "validator_worker"
    else:
        next_step = "FINISH"

    print(f"   >> Supervisor Decision: Routing to '{next_step}'")
    return {"next_agent": next_step}


def math_worker_node(state: TeamState):
    print("🔬 [Math Worker Desk (Mistral)] Processing numbers with precision tool...")

    # Force Mistral to acknowledge its specific calculation role
    worker_prompt = (
        "You are the specialized Math Worker. Use your calculate_sharpe_ratio tool "
        "to compute the precise metric. Do not try to guess or do the math in text."
    )

    messages = [HumanMessage(content=worker_prompt)] + state["messages"]
    response = math_llm.invoke(messages)

    # Capture tool output or standard text generation safely
    return {"messages": [AIMessage(content=f"Math Worker final log: {response.content}")]}


def validator_worker_node(state: TeamState):
    print("⚖️ [Validator Desk (Gemma 2)] Checking data parameters...")

    validation_prompt = (
        "You are the Risk Validator. Analyze the calculated metrics in the history. "
        "Provide a short, one-sentence quantitative evaluation stating if this Sharpe Ratio "
        "represents healthy risk-adjusted performance."
    )

    messages = state["messages"] + [HumanMessage(content=validation_prompt)]
    response = validator_llm.invoke(messages)
    return {"messages": [AIMessage(content=f"Validator final log: {response.content}")]}

# 5. ASSEMBLING THE WORKFLOW PIPES

builder = StateGraph(TeamState)

# Define our desks
builder.add_node("supervisor", supervisor_node)
builder.add_node("math_worker", math_worker_node)
builder.add_node("validator_worker", validator_worker_node)

# Set the entrance route
builder.set_entry_point("supervisor")

# Define the automatic loop backbones: Workers always return to the Supervisor
builder.add_edge("math_worker", "supervisor")
builder.add_edge("validator_worker", "supervisor")

# THE DYNAMIC TRAFFIC COP: Read state["next_agent"] to choose the next node direction
def router_logic(state: TeamState):
    if state["next_agent"] == "math_worker":
        return "math_worker"
    elif state["next_agent"] == "validator_worker":
        return "validator_worker"
    else:
        return "__end__"

builder.add_conditional_edges(
    "supervisor",
    router_logic,
    {
        "math_worker": "math_worker",
        "validator_worker": "validator_worker",
        "__end__": END
    }
)

graph = builder.compile()

# 6. RUN THE TEAM TEST

print("\n--- STARTING LOCAL MULTI-AGENT TEAM UP ---")
query = "Can you check this portfolio performance? Return is 14%, volatility is 18%, and risk-free is 3%."

inputs = {
    "messages" : [HumanMessage(content=query)],
    "next_agent" : ""
}

for output in graph.stream(inputs, stream_mode="updates"):
    for node, value in output.items():
        print(f"--> Done with Node : '{node}'")
        # If the node appended a message to the state folder, extract and print it!
        if "messages" in value:
            print(f"    Message Content: {value['messages'][-1].content}\n")
