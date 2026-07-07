import os
from dotenv import load_dotenv
from langchain_classic.agents import create_react_agent
from typing import TypedDict, Dict, Annotated, Any
from operator import add
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
# from langchain.agents import create_agent
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama

load_dotenv()

# 1. THE PRODUCTION MULTI-VARIABLE STATE

class ProductionState(TypedDict):

    messages: Annotated[list[BaseMessage], add]

    portfolio_metrics: Dict[str, Any] # Isolated memory channel for numbers

    remaining_steps: int # Required by LangGraph's safety manager

# 2. THE SYSTEM TOOL (Mutates state data while calculating)

@tool

def calculate_and_save_sharpe(portfolio_return : float, risk_free_rate : float, standard_deviation : float) -> str:
    """
    Calculates the Sharpe Ratio and returns a summary.
    Use this tool whenever a user asks for risk metrics or portfolio calculations.
    """
    print(f"\n⚡ [SYSTEM TOOL] Executing Python math tool...")

    # Run precision math
    sharpe = (portfolio_return - risk_free_rate) / standard_deviation

    # We return a string summary that the model reads
    return f"Calculation executed successfully. Resulting Sharpe Ratio is: {sharpe:.4f}"

tools_list = [calculate_and_save_sharpe]

# 3. INITIALIZE LOCAL LLM & COMPILE THE AGENT

llm = ChatOllama(
    model="llama3.1:latest",
    temperature=0.0,
    base_url="http://localhost:11434",
    keep_alive='5s'
)

memory_cabinet = MemorySaver()

# Force the local agent to act as a disciplined analyst
system_prompt = (
    "You are a strict quantitative analyst. "
    "Always rely on your calculation tools to resolve mathematical queries. "
    "Do not guess or try to compute metrics yourself."
)

agent_graph = create_react_agent(
    model = llm,
    tools = tools_list,
    checkpointer = memory_cabinet,
    prompt = system_prompt,
    state_schema = ProductionState
)

# 4. EXECUTION

config = {"configurable" : {"thread_id" : "production_session_51"}}

print("\n--- STARTING PRODUCTION MULTI-VARIABLE AGENT RUN ---")
query = "Can you check my portfolio performance? Return is 14%, volatility is 18%, and the risk-free rate is 3%."

# Initialize our state inputs matching the schema keys exactly

inputs = {
    "messages" : [HumanMessage(content=query)],
    "portfolio_metrics" : {"last_updated_by" : "user_query", "status" : "PENDING"}
}

for output in agent_graph.stream(inputs, config=config, stream_mode = "updates"):
    for node, value in output.items():
        print(f"--> Finished Node: '{node}")
        if "messages" in value:
            print(f"    Last Message : {value['messages'][-1].content}\n")
