import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal, Optional
from operator import add
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama

load_dotenv()


# =====================================================================
# 1. ENTERPRISE STATE (Explicit Tracking Keys)
# =====================================================================
class PortfolioMetrics(BaseModel):
    sharpe_ratio: Optional[float] = None
    is_healthy: Optional[bool] = None


class TeamState(TypedDict):
    messages: Annotated[list[BaseMessage], add]
    next_agent: str
    # Explicit schema storage separating data variables from conversational text
    metrics: PortfolioMetrics


# =====================================================================
# 2. LOCAL CALCULATION ENGINE
# =====================================================================
def calculate_sharpe_ratio(portfolio_return: float, risk_free_rate: float, standard_deviation: float) -> float:
    print(f"\n⚡ [SYSTEM MATH ENGINE] Firing local Python function...")
    return (portfolio_return - risk_free_rate) / standard_deviation


sharpe_tool_object = StructuredTool.from_function(
    func=calculate_sharpe_ratio,
    name="calculate_sharpe_ratio",
    description="Calculates the Sharpe Ratio of an investment portfolio."
)

tools_list = [sharpe_tool_object]
tool_executor_node = ToolNode(tools_list)


# =====================================================================
# 3. ROUTING SCHEMA & MODELS
# =====================================================================
class RouterSchema(BaseModel):
    next_agent: Literal["math_worker", "validator_worker", "FINISH"] = Field(
        description="Route based on execution completeness."
    )


raw_llm = ChatOllama(model="llama3.1:latest", temperature=0.0, base_url="http://localhost:11434")
supervisor_llm = raw_llm.with_structured_output(RouterSchema)
math_llm = ChatOllama(model="mistral:latest", temperature=0.0, base_url="http://localhost:11434").bind_tools(tools_list)
validator_llm = ChatOllama(model="gemma2:2b", temperature=0.0, base_url="http://localhost:11434")


# =====================================================================
# 4. OPTIMIZED WORKER NODES
# =====================================================================
def supervisor_node(state: TeamState):
    print("\n👑 [Supervisor (Llama 3.1)] Assessing explicit state keys...")

    # Check programmatic data attributes directly instead of string parsing message history logs
    has_tool_run = state["metrics"].sharpe_ratio is not None
    has_validation = state["metrics"].is_healthy is not None

    print(f"   [STATE DATA] Sharpe Ratio Saved: {state['metrics'].sharpe_ratio} | Validated: {has_validation}")

    if has_tool_run and has_validation:
        print("   >> Hard Exit Guard Triggered: Pipeline data criteria complete.")
        return {"next_agent": "FINISH"}

    prompt = (
        "You are the Portfolio Supervisor Manager. Direct the routing flow:\n"
        f"- Is Math Completed? {has_tool_run}\n"
        f"- Is Validation Completed? {has_validation}\n\n"
        "Conditions: Math False -> 'math_worker'. Math True and Validation False -> 'validator_worker'."
    )

    messages = [state["messages"][0], HumanMessage(content=prompt)]
    response: RouterSchema = supervisor_llm.invoke(messages)
    return {"next_agent": response.next_agent}


def math_worker_node(state: TeamState):
    print("🔬 [Math Worker (Mistral)] Routing parameter inputs...")
    worker_prompt = "Extract parameters and call the calculate_sharpe_ratio tool immediately."
    messages = [HumanMessage(content=worker_prompt), state["messages"][0]]
    response = math_llm.invoke(messages)
    return {"messages": [response]}


# =====================================================================
# 4. OPTIMIZED WORKER NODES & UPDATERS
# =====================================================================
# We rewrite the edge function into a proper State Interceptor Node
def state_updater_node(state: TeamState):
    print("\n📊 [STATE INTERCEPTOR] Processing raw tool data into DB...")
    last_msg = state["messages"][-1]
    current_metrics = state.get("metrics", PortfolioMetrics())

    if last_msg.__class__.__name__ == "ToolMessage" or getattr(last_msg, "type", "") == "tool":
        try:
            val = float(str(last_msg.content))
            current_metrics.sharpe_ratio = round(val, 4)
            print(f"   Stored Sharpe Ratio: {current_metrics.sharpe_ratio}")
        except ValueError:
            pass

    # Return the updated dictionary to modify the global graph state
    return {"metrics": current_metrics}


def validator_worker_node(state: TeamState):
    print("⚖️ [Validator (Gemma 2)] Inspecting structured state metrics...")

    # Read directly from structured state fields rather than parsing long chat transcripts
    ratio = state["metrics"].sharpe_ratio

    prompt = (
        f"The system calculated an exact Sharpe Ratio of {ratio}.\n"
        "Provide a short, professional one-sentence financial evaluation.\n"
        "Your response MUST start with: 'Validator Final Review:'"
    )

    response = validator_llm.invoke([HumanMessage(content=prompt)])

    # Mark the validation tracking schema as complete
    updated_metrics = state["metrics"]
    updated_metrics.is_healthy = True

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
builder.add_node("tools", tool_executor_node)
builder.add_node("state_updater", state_updater_node)
builder.add_node("validator_worker", validator_worker_node)

builder.set_entry_point("supervisor")

builder.add_conditional_edges("supervisor", lambda state: state["next_agent"], {
    "math_worker": "math_worker",
    "validator_worker": "validator_worker",
    "FINISH": END
})

# After the tools node executes, intercept the state flow to run our data extraction helper
builder.add_conditional_edges("math_worker",
                              lambda state: "tools" if (
                                          hasattr(state["messages"][-1], "tool_calls") and state["messages"][
                                      -1].tool_calls) else "supervisor",
                              {"tools": "tools", "supervisor": "supervisor"}
                              )

# 3. CRITICAL CORRECTION: Tools node goes straight to our updater node
builder.add_edge("tools", "state_updater")

# 4. State updater finishes writing to database, goes back to supervisor
builder.add_edge("state_updater", "supervisor")

# 5. Validator goes back to supervisor
builder.add_edge("validator_worker", "supervisor")

graph = builder.compile()

# =====================================================================
# 6. RUN THE SYSTEM
# =====================================================================
print("\n--- STARTING KULIGIN-STRUCTURED QUANT MULTI-AGENT RUN ---")
query = "Can you check my portfolio performance? Return is 14%, volatility is 18%, and the risk-free rate is 3%."

inputs = {
    "messages": [HumanMessage(content=query)],
    "next_agent": "",
    "metrics": PortfolioMetrics()
}

for output in graph.stream(inputs, stream_mode="updates"):
    for node, value in output.items():
        print(f"--> Done with Node: '{node}'")
