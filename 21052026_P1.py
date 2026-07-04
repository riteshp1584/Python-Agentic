import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from pyexpat.errors import messages

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

# Put our tools inside an accessible list
tools_list =  [calculate_sharpe_ratio]

# 2. INITIALIZE THE LOCAL BRAIN (Using your llama3.1:latest)

llm = ChatOllama(
    model="llama3.1:latest", # <-- MATCHES OUR OLLAMA LIST EXACTLY
    temperature=0.0, # Keep it deterministic for reliable math tool calls
    base_url="http://localhost:11434"
)

# 3. COMPILE THE REACTION AGENT WITH MEMORY

memory_cabinet = MemorySaver()

agent_graph = create_agent(
    model = llm,
    tools = tools_list,
    checkpointer = memory_cabinet
)

# 4. RUN THE LOCAL TEST

config = {"configurable" : {"thread_id" : "local_quant_session"}}

print("\n--- STARTING LOCAL OLLAMA REACT AGENT RUN ---")
query = "Can you check my portfolio? Return is 14%, volatility is 18%, and the risk-free rate is 3%."

inputs = {"messages" : [("user", query)]}

# We stream the nodes step-by-step so you can watch your local machine process it
for output in agent_graph.stream(inputs, config=config, stream_mode="updates"):
    for node, value in output.items():
        print(f"--> Local Finished Node: '{node}'")
        if "messages" in value:
            print(f"    Message Content: {value['messages'][-1].content}\n")
