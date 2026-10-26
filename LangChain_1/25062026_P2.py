import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal, Optional, Dict
from operator import add
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.messages import trim_messages, RemoveMessage

load_dotenv()

# =====================================================================
# 1. KULIGIN EXPANDED STATE SCHEMA
# =====================================================================
class PortfolioMetrics(BaseModel):
    sharpe_ratio: Optional[float] = None
    is_healthy: Optional[bool] = None
    optimization_required: Optional[bool] = None
    optimized_weights: Optional[Dict[str, str]] = None
    # RAG Context tracking slot
    retrieved_mandate: Optional[str] = None


class TeamState(TypedDict):
    messages: Annotated[list[BaseMessage], add]
    next_agent: str
    metrics: PortfolioMetrics

# =====================================================================
# 2. IN-MEMORY VECTOR POLICY DATABASE (Chapter 4 Core)
# =====================================================================
print("📦 Initializing Local Risk Mandate Vector Database...")

# Using your existing local Llama 3.1 to generate vector embeddings out-of-the-box
embeddings = OllamaEmbeddings(model="llama3.1:latest", base_url="http://localhost:11434")
vector_store = InMemoryVectorStore(embeddings)

# Synthetic Corporate Mandate Documents
policies = [
    Document(
        page_content="Aggressive Growth Strategy Mandate: This portfolio targets high capital appreciation. Minimum acceptable Sharpe Ratio threshold is 0.55. Any portfolio falling below 0.55 must be flagged for immediate weight optimization.",
        metadata={"strategy": "aggressive"}
    ),
    Document(
        page_content="Institutional Conservative Mandate: This portfolio services low-risk corporate clients. Minimum acceptable Sharpe Ratio threshold is 0.85. Portfolios with a Sharpe Ratio less than 0.85 require immediate rebalancing into safe fixed-income assets.",
        metadata={"strategy": "conservative"}
    ),
    Document(
        page_content="Standard Balanced Fund Mandate: This portfolio maintains a typical 60/40 style risk profile. Minimum acceptable Sharpe Ratio threshold is 0.70. Optimization triggers automatically if performance metrics fall below this line.",
        metadata={"strategy": "balanced"}
    )
]

# Synchronously load documents into memory
vector_store.add_documents(policies)
print("✅ Vector Database successfully populated with 3 corporate investment strategies.")

# =====================================================================
# 3. TOOLS & MODELS SETUP
# =====================================================================
def calculate_sharpe_ratio(portfolio_return: float, risk_free_rate: float, standard_deviation: float) -> float:
    print(f"\n⚡ [SYSTEM MATH ENGINE] Firing local calculation function...")
    return (portfolio_return - risk_free_rate) / standard_deviation


def optimize_portfolio_weights(current_sharpe: float) -> str:
    print(f"\n⚡ [OPTIMIZATION ENGINE] Executing asset rebalancing matrix...")
    return "Equities: 50%, Fixed Income: 40%, Alternatives: 10%"


sharpe_tool = StructuredTool.from_function(func=calculate_sharpe_ratio, name="calculate_sharpe_ratio",
                                           description="Calculates Sharpe Ratio.")
optimize_tool = StructuredTool.from_function(func=optimize_portfolio_weights, name="optimize_portfolio_weights",
                                             description="Optimizes portfolio weights.")

tools_list = [sharpe_tool, optimize_tool]
tool_executor_node = ToolNode(tools_list)

math_llm = ChatOllama(model="mistral:latest", temperature=0.0, base_url="http://localhost:11434").bind_tools(
    [sharpe_tool])
optimizer_llm = ChatOllama(model="mistral:latest", temperature=0.0, base_url="http://localhost:11434").bind_tools(
    [optimize_tool])
validator_llm = ChatOllama(model="gemma2:2b", temperature=0.0, base_url="http://localhost:11434")

# Create a sliding window memory manager
message_trimmer = trim_messages(
    max_tokens=3, # Low count forces it to look at message quantity rather than massive token lengths
    strategy="last",
    token_counter=len, # Counts raw message items in the list
    include_system=True, # Prevent discarding core system rules
    allow_partial=False,
    start_on="human" # Always ensure the context starts cleanly with a human message
)


# =====================================================================
# 4. REFACTORED AGENT WORKER NODES
# =====================================================================
def supervisor_node(state: TeamState):
    print("\n👑 [Supervisor (Code Router)] Assessing explicit state keys...")

    has_math = state["metrics"].sharpe_ratio is not None
    has_rag = state["metrics"].retrieved_mandate is not None
    has_validation = state["metrics"].is_healthy is not None
    needs_opt = state["metrics"].optimization_required
    has_opt = state["metrics"].optimized_weights is not None

    print(
        f"   [STATE] Math: {has_math} | RAG Context: {has_rag} | Validated: {has_validation} | Needs Opt: {needs_opt}")

    if not has_math:
        next_step = "math_worker"
    elif has_math and not has_rag:
        next_step = "rag_retrieval_worker"  # Dynamic step inserted between Math and Validation
    elif has_rag and not has_validation:
        next_step = "validator_worker"
    elif has_validation and needs_opt and not has_opt:
        next_step = "optimizer_worker"
    else:
        next_step = "FINISH"

    print(f"   >> Programmatic Routing Decision: next_agent='{next_step}'")
    return {"next_agent": next_step}


def math_worker_node(state: TeamState):
    print("🔬 [Math Worker (Mistral)] Pruning memory window and routing parameters...")

    # 1. Trim down historical clutter
    optimized_history = message_trimmer.invoke(state["messages"])

    worker_prompt = "Extract parameters and call the calculate_sharpe_ratio tool immediately."

    # 2. Combine the streamlined history with the active node instruction
    messages = [HumanMessage(content=worker_prompt)] + optimized_history

    response = math_llm.invoke(messages)
    return {"messages": [response]}

def rag_retrieval_worker_node(state: TeamState):
    print("🔍 [RAG Context Worker] Querying in-memory policy vector database...")
    user_query = state["messages"][0].content

    # Semantic search to match user intent ("balanced", "conservative", "aggressive")
    search_results = vector_store.similarity_search(user_query, k=1)
    matched_policy = search_results[0].page_content

    print(f"   [RAG HIT] Successfully retrieved matching policy rule framework.")

    updated_metrics = state["metrics"]
    updated_metrics.retrieved_mandate = matched_policy
    return {"metrics": updated_metrics}

def validator_worker_node(state: TeamState):
    print("⚖️ [Validator (Gemma 2)] Grounding analysis against retrieved RAG mandate...")
    ratio = state["metrics"].sharpe_ratio
    mandate = state["metrics"].retrieved_mandate

    # We pass the RAG mandate text directly into Gemma's evaluation prompt
    prompt = (
        f"You are an institutional risk compliance evaluator.\n"
        f"The calculated portfolio Sharpe Ratio is: {ratio}\n"
        f"The active corporate strategy mandate rules state:\n\"{mandate}\"\n\n"
        "Task: Based strictly on the mandate rule text provided, is this Sharpe Ratio acceptable?\n"
        "If it is below the mandated threshold, determine if optimization is required.\n"
        "Your final response MUST conclude with exactly either: 'OPTIMIZATION_REQUIRED: TRUE' or 'OPTIMIZATION_REQUIRED: FALSE'."
    )

    response = validator_llm.invoke([HumanMessage(content=prompt)])
    response_text = response.content.upper()

    opt_flag = "OPTIMIZATION_REQUIRED: TRUE" in response_text

    updated_metrics = state["metrics"]
    updated_metrics.is_healthy = True
    updated_metrics.optimization_required = opt_flag

    print(f"   [VALIDATOR ANALYSIS] RAG-Grounded Audit Complete. Optimization required: {opt_flag}")
    return {
        "messages": [AIMessage(content=f"Validator RAG Review: {response.content}")],
        "metrics": updated_metrics
    }


def optimizer_worker_node(state: TeamState):
    print("🚀 [Optimizer Worker (Mistral)] Pruning memory window and re-allocating assets...")

    # Trim down historical clutter so old tool calls don't bleed into the prompt
    optimized_history = message_trimmer.invoke(state["messages"])

    ratio = state["metrics"].sharpe_ratio

    worker_prompt = f"Portfolio sub-optimal at {ratio}. Execute optimize_portfolio_weights tool immediately."

    messages = [HumanMessage(content=worker_prompt)] + optimized_history
    response = optimizer_llm.invoke(messages)
    return {"messages": [response]}

def state_updater_node(state: TeamState):
    print("📊 [STATE INTERCEPTOR] Processing raw tool outputs into Fact Table...")
    last_msg = state["messages"][-1]
    current_metrics = state.get("metrics", PortfolioMetrics())

    if last_msg.__class__.__name__ == "ToolMessage" or getattr(last_msg, "type", "") == "tool":
        content_str = str(last_msg.content)
        if "0." in content_str or "." in content_str and len(content_str) < 20:
            try:
                current_metrics.sharpe_ratio = round(float(content_str), 4)
                print(f"   Stored Sharpe Ratio: {current_metrics.sharpe_ratio}")
            except ValueError:
                pass
        elif "Equities" in content_str:
            current_metrics.optimized_weights = {"allocations": content_str}
            print(f"   Stored Optimized Weights: {current_metrics.optimized_weights}")

    return {"metrics": current_metrics}


# =====================================================================
# 5. GRAPH ASSEMBLY WITH RAG CONNECTIONS
# =====================================================================
builder = StateGraph(TeamState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("math_worker", math_worker_node)
builder.add_node("rag_retrieval_worker", rag_retrieval_worker_node)  # Node registered
builder.add_node("optimizer_worker", optimizer_worker_node)
builder.add_node("tools", tool_executor_node)
builder.add_node("state_updater", state_updater_node)
builder.add_node("validator_worker", validator_worker_node)

builder.set_entry_point("supervisor")

builder.add_conditional_edges("supervisor", lambda state: state["next_agent"], {
    "math_worker": "math_worker",
    "rag_retrieval_worker": "rag_retrieval_worker",
    "validator_worker": "validator_worker",
    "optimizer_worker": "optimizer_worker",
    "FINISH": END
})


def tool_routing_logic(state: TeamState):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "supervisor"


builder.add_conditional_edges("math_worker", tool_routing_logic, {"tools": "tools", "supervisor": "supervisor"})
builder.add_conditional_edges("optimizer_worker", tool_routing_logic, {"tools": "tools", "supervisor": "supervisor"})

builder.add_edge("tools", "state_updater")
builder.add_edge("state_updater", "supervisor")
builder.add_edge("rag_retrieval_worker", "supervisor")
builder.add_edge("validator_worker", "supervisor")

graph = builder.compile()

# =====================================================================
# 6. EXECUTE ADVANCED INTENSIFIED RUN
# =====================================================================
print("\n--- STARTING CHAPTER 4 INTELLIGENT RAG MULTI-AGENT RUN ---")

# Notice we explicitly ask for the "Institutional Conservative Mandate" in our query!
query = "Check my client's portfolio performance under our Institutional Conservative Mandate rules. Return is 10%, volatility is 12%, and risk-free is 2%."

inputs = {
    "messages": [HumanMessage(content=query)],
    "next_agent": "",
    "metrics": PortfolioMetrics()
}

for output in graph.stream(inputs, stream_mode="updates"):
    for node, value in output.items():
        print(f"--> Done with Node: '{node}'")

# print("Check:")
# print(state["messages"])
