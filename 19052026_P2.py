import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

# NEW IMPORT: This completely replaces manual node wiring and conditional edge setups!
# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
load_dotenv()

# 1. THE TOOL (The Calculator on the Desk | Remains Exactly the Same)
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

# 2. THE BRAIN & THE PREBUILT AGENT
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

# We initialize a MemorySaver cabinet so our agent has a chat history log
memory_cabinet = MemorySaver()

# Instead of manually writing builder = StateGraph(), adding nodes, and
# mapping conditional edges, we let LangGraph compile the entire structure for us:
agent_graph = create_agent(
    model = llm,
    tools = tools_list,
    checkpointer = memory_cabinet
)

# 3. TEST RUN WITH MEMORY PASSPORT (Thread ID)
config = {"configurable" : {"thread_id" : "portfolio_analysis_session"}}

print("\n--- STARTING PREBUILT REACT AGENT RUN ---")
query = "Can you calculate my Sharpe ratio? My return is 14%, volatility is 18%, and the risk-free rate is 3%."

inputs = {"messages" : [("user", query)]}

# Execute using stream to watch the prebuilt nodes run step-by-step
for output in agent_graph.stream(inputs, config=config, stream_mode="updates"):
    for node, value in output.items():
        print(f"--> Prebuilt Finished Node: '{node}'")
        # Let's print out what the agent is actually thinking or saying at this node step
        if "messages" in value:
            print(f'    Message Content: {value["messages"][-1].content}\n')