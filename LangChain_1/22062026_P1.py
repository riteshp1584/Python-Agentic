import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal, Optional, Dict
from operator import add
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama

load_dotenv()

# =====================================================================
# 1. EXPANDED ENTERPRISE STATE (Tracking Allocation Weights)
# =====================================================================

class PortfolioMetrics(BaseModel):
    sharpe_ratio: Optional[float] = None
    is_healthy: Optional[bool] = None
    optimization_required: Optional[bool] = None
    # Explicitly store optimal asset weight percentages as native types
    optimized_weights: Optional[Dict[str, str]] = None

class TeamState(TypedDict):
    messages: Annotated[list[BaseMessage], add]
    next_agent: str
    metrics: PortfolioMetrics

# =====================================================================
# 2. CALCULATION & OPTIMIZATION TOOLS
# =====================================================================

def calculate_sharpe_ratio(portfolio_return: float, risk_free_rate: float, standard_deviation: float) -> float:
    print(f"\n⚡ [SYSTEM MATH ENGINE] Firing local calculation function...")
    return (portfolio_return - risk_free_rate) / standard_deviation

def optimize_portfolio_weights(current_sharpe: float) -> str:
    print(f"\n⚡ [OPTIMIZATION ENGINE] Executing asset rebalancing matrix...")
    # Simulating a basic allocation rebalance tool to boost performance past the threshold
    return "Equities: 60%, Fixed Income: 30%, Alternatives: 10%"

sharpe_tool = StructuredTool.from_function(func=calculate_sharpe_ratio,
                                           name="calculate_sharpe_ratio",
                                           description="Calculates the Sharpe Ratio.")

optimize_tool = StructuredTool.from_function(func=optimize_portfolio_weights,
                                             name="optimize_portfolio_weights",
                                             description="Optimizes asset allocation weights.")

tools_list = [sharpe_tool, optimize_tool]
tool_executor_node = ToolNode(tools_list)

# =====================================================================
# 3. INITIALIZE MODELS
# =====================================================================

math_llm = ChatOllama(model="mistral:latest", temperature=0.0, base_url="http://localhost:11434").bind_tools([sharpe_tool])
optimizer_llm  = ChatOllama(model="mistral:latest", temperature=0.0, base_url="http://localhost:11434").bind_tools([optimize_tool])
validator_llm  = ChatOllama(model="gemma2:2b", temperature=0.0, base_url="http://localhost:11434")

# =====================================================================
# 4. DETERMINISTIC WORKER NODES
# =====================================================================

def supervisor_node(state: TeamState):
    print("\n👑 [Supervisor (Code Router)] Assessing explicit state keys...")

    has_math = state["metrics"].sharpe_ratio is not None
    has_validation = state["metrics"].is_healthy is not None
    needs_opt = state["metrics"].optimization_required
    has_opt = state["metrics"].optimized_weights is not None

    print(f"   [STATE] Math: {has_math} | Validated: {has_validation} | Needs Opt: {needs_opt} | Opt Complete: {has_opt}")

    # Strict deterministic routing logic tree
    if not has_math:
        next_step = "math_worker"
    elif has_math and not has_validation:
        next_step = "validator_worker"
    elif has_validation and needs_opt and not has_opt:
        next_step = "optimizer_worker"
    else:
        next_step = "FINISH"

    print(f"   >> Programmatic Routing Decision: next_agent='{next_step}'")

    return {"next_agent": next_step}

def math_worker_node(state: TeamState):
    print("🔬 [Math Worker (Mistral)] Routing parameter inputs...")
    worker_prompt = "Extract parameters and call the calculate_sharpe_ratio tool immediately."
    messages = [HumanMessage(content=worker_prompt), state["messages"][0]]
    response = math_llm.invoke(messages)
    return {"messages": [response]}

def optimizer_worker_node(state: TeamState):
    print("🚀 [Optimizer Worker (Mistral)] Re-allocating asset classes...")
    ratio = state["metrics"].sharpe_ratio
    worker_prompt = (f"The portfolio is sub-optimal with a Sharpe Ratio of {ratio}."
                     f"Run the optimize_portfolio_weights tool immediately to resolve this.")
    response = optimizer_llm.invoke([HumanMessage(content=worker_prompt)])
    return {"messages": [response]}

# Updated data interceptor to catch BOTH tool types cleanly
def state_updater_node(state: TeamState):
    print("📊 [STATE INTERCEPTOR] Processing raw tool outputs into Fact Table...")
    last_msg = state["messages"][-1]
    current_metrics = state.get("metrics", PortfolioMetrics())

    if last_msg.__class__.__name__ == "ToolMessage" or getattr(last_msg, "type", "") == "tool":
        content_str = str(last_msg.content)

        # If it came from the Sharpe calculator
        if "0." in content_str or "." in content_str and len(content_str) < 20:
            try:
                current_metrics.sharpe_ratio = round(float(content_str), 4)
                print(f"   Stored Sharpe Ratio: {current_metrics.sharpe_ratio}")
            except ValueError:
                pass
        # If it came from the Optimizer tool
        elif "Equities" in content_str:
            current_metrics.optimized_weights = {"allocations": content_str}
            print(f"   Stored Optimized Weights: {current_metrics.optimized_weights}")

    return {"metrics": current_metrics}

def validator_worker_node(state: TeamState):
    print("⚖️ [Validator (Gemma 2)] Inspecting structured state metrics...")
    ratio = state["metrics"].sharpe_ratio

    # We set a threshold: anything under 0.75 triggers optimization requirement
    opt_flag = True if ratio < 0.75 else False

    prompt = (
        f"The system calculated an exact Sharpe Ratio of {ratio}.\n"
        "Provide a short, professional one-sentence financial evaluation.\n"
        "Your response MUST start with: 'Validator Final Review:'"
    )
    response = validator_llm.invoke([HumanMessage(content=prompt)])

    updated_metrics = state["metrics"]
    updated_metrics.is_healthy = True
    updated_metrics.optimization_required = opt_flag

    print(f"   [VALIDATOR ANALYSIS] Threshold Target Met: {not opt_flag} | Optimization Flagged: {opt_flag}")

    return {
        "messages": [AIMessage(content=f"Validator Final Review: {response.content}")],
        "metrics": updated_metrics
    }

# =====================================================================
# 5. GRAPH ASSEMBLY
# =====================================================================

builder = StateGraph(TeamState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("math_worker", math_worker_node)
builder.add_node("optimizer_worker", optimizer_worker_node)
builder.add_node("tools", tool_executor_node)
builder.add_node("state_updater", state_updater_node)
builder.add_node("validator_worker", validator_worker_node)

builder.set_entry_point("supervisor")

builder.add_conditional_edges("supervisor", lambda state: state["next_agent"], {
    "math_worker": "math_worker",
    "validator_worker": "validator_worker",
    "optimizer_worker": "optimizer_worker",
    "FINISH": END
})

# Dynamic Router to check if Math Worker or Optimizer Worker invoked a tool
def tool_routing_logic(state: TeamState):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "supervisor"

builder.add_conditional_edges("math_worker", tool_routing_logic, {"tools": "tools", "supervisor": "supervisor"})
builder.add_conditional_edges("optimizer_worker", tool_routing_logic, {"tools": "tools", "supervisor": "supervisor"})

builder.add_edge("tools", "state_updater")
builder.add_edge("state_updater", "supervisor")
builder.add_edge("validator_worker", "supervisor")

graph = builder.compile()

# =====================================================================
# 6. EXECUTE SYSTEM
# =====================================================================

print("\n--- STARTING ADVANCED MULTI-AGENT OPTIMIZATION RUN ---")
query = "Can you check my portfolio performance? Return is 14%, volatility is 18%, and the risk-free rate is 3%."

inputs = {
    "messages": [HumanMessage(content=query)],
    "next_agent": "",
    "metrics": PortfolioMetrics()
}

for output in graph.stream(inputs, stream_mode="updates"):
    for node, value in output.items():
        print(f"--> Done with Node: '{node}'")
