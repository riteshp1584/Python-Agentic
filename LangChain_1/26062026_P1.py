import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal, Optional, Dict
from operator import add
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.documents import Document

load_dotenv()


# =====================================================================
# 1. PURE-PYTHON VECTOR STORE FALLBACK (CRASH-PROOF RAG)
# =====================================================================
class RobustLocalVectorStore:
    """
    A lightweight, pure-Python in-memory search store that mimics LangChain's
    vector store interface. Completely eliminates dependency on external embedding services.
    """

    def __init__(self):
        self.documents = []

    def add_documents(self, documents: list[Document]):
        self.documents.extend(documents)
        print(f"   [VECTOR DB] Successfully indexed {len(documents)} policy documents.")

    def similarity_search(self, query: str, k: int = 1) -> list[Document]:
        # Normalize and tokenize the user query
        query_tokens = set(query.lower().replace(",", "").replace(".", "").split())
        scored_documents = []

        for doc in self.documents:
            # Tokenize document content
            doc_tokens = set(doc.page_content.lower().replace(",", "").replace(".", "").split())
            # Calculate overlapping words (intersection) to score relevance
            overlap = len(query_tokens.intersection(doc_tokens))
            scored_documents.append((overlap, doc))

        # Sort documents dynamically by relevance score (highest overlap first)
        scored_documents.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_documents[:k]]


# Initialize our zero-dependency policy database
print("📦 Initializing Local Risk Mandate Policy Database...")
vector_store = RobustLocalVectorStore()

# Injecting the synthetic investment strategies
policies = [
    Document(
        page_content="Institutional Conservative Mandate: This portfolio services low-risk corporate clients. Minimum acceptable Sharpe Ratio threshold is 0.85. Portfolios with a Sharpe Ratio less than 0.85 require immediate rebalancing into safe fixed-income assets.",
        metadata={"strategy": "conservative"}
    ),
    Document(
        page_content="Aggressive Growth Strategy Mandate: This portfolio targets high capital appreciation. Minimum acceptable Sharpe Ratio threshold is 0.55. Any portfolio falling below 0.55 must be flagged for immediate weight optimization.",
        metadata={"strategy": "aggressive"}
    ),
    Document(
        page_content="Standard Balanced Fund Mandate: This portfolio maintains a typical 60/40 style risk profile. Minimum acceptable Sharpe Ratio threshold is 0.70. Optimization triggers automatically if performance metrics fall below this line.",
        metadata={"strategy": "balanced"}
    )
]
vector_store.add_documents(policies)


# =====================================================================
# 2. STATE DEFINITIONS & REDUCERS
# =====================================================================
class PortfolioMetrics(BaseModel):
    sharpe_ratio: Optional[float] = None

    # Parallel audit tracking keys
    risk_validation_complete: bool = False
    optimization_required: Optional[bool] = None

    compliance_audit_complete: bool = False
    compliance_passed: Optional[bool] = None

    optimized_weights: Optional[Dict[str, str]] = None
    retrieved_mandate: Optional[str] = None


def merge_metrics(left: PortfolioMetrics, right: PortfolioMetrics) -> PortfolioMetrics:
    """
    State Reducer: Merges two PortfolioMetrics updates.
    Ensures parallel execution nodes can update distinct properties without collision.
    Takes whichever PortfolioMetrics is non-empty; left or right
    """
    if left is None:
        return right
    if right is None:
        return left

    left_dict = left.model_dump()
    right_dict = right.model_dump()

    for key, val in right_dict.items():
        # Carry over boolean updates that transitioned to True
        if key in ["risk_validation_complete", "compliance_audit_complete"]:
            left_dict[key] = left_dict[key] or right_dict[key]
        # Carry over other metrics if they are explicitly updated
        elif val is not None:
            left_dict[key] = val

    return PortfolioMetrics(**left_dict)


class TeamState(TypedDict):
    messages: Annotated[list[BaseMessage], add]
    # List of parallel agents slated for simultaneous execution
    next_agents: list[str]
    # Configured with our state reducer to enable collision-free parallel writes
    metrics: Annotated[PortfolioMetrics, merge_metrics]


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

tool_executor_node = ToolNode([sharpe_tool, optimize_tool])

# Bind models locally
math_llm = ChatOllama(model="mistral:latest", temperature=0.0, base_url="http://localhost:11434").bind_tools(
    [sharpe_tool])
optimizer_llm = ChatOllama(model="mistral:latest", temperature=0.0, base_url="http://localhost:11434").bind_tools(
    [optimize_tool])
validator_llm = ChatOllama(model="gemma2:2b", temperature=0.0, base_url="http://localhost:11434")
compliance_llm = ChatOllama(model="mistral:latest", temperature=0.0, base_url="http://localhost:11434")


# =====================================================================
# 4. REINFORCED WORKER NODES
# =====================================================================
def supervisor_node(state: TeamState):
    print("\n👑 [Supervisor (Parallel Coordinator)] Assessing graph convergence...")

    metrics = state["metrics"]
    has_math = metrics.sharpe_ratio is not None
    has_rag = metrics.retrieved_mandate is not None

    risk_done = metrics.risk_validation_complete
    comp_done = metrics.compliance_audit_complete

    # 1. First step: Process math calculations
    if not has_math:
        return {"next_agents": ["math_worker"]}

    # 2. Second step: Fetch RAG context documentation
    if has_math and not has_rag:
        return {"next_agents": ["rag_retrieval_worker"]}

    # 3. FAN-OUT: Run Risk and Compliance validation in parallel
    if has_rag and not (risk_done and comp_done):
        targets = []
        if not risk_done: targets.append("validator_worker")
        if not comp_done: targets.append("compliance_worker")
        print(f"   🔀 TRIGGERING FAN-OUT: Spawning concurrent execution for {targets}")
        return {"next_agents": targets}

    # 4. FAN-IN MERGE: Assess results before optimizing
    needs_opt = metrics.optimization_required or (metrics.compliance_passed == False)
    has_opt = metrics.optimized_weights is not None

    if needs_opt and not has_opt:
        return {"next_agents": ["optimizer_worker"]}

    print("   🏁 All parallel validation gates passed successfully.")
    return {"next_agents": ["FINISH"]}


def math_worker_node(state: TeamState):
    print("🔬 [Math Worker (Mistral)] Processing parameters...")
    messages = [HumanMessage(content="Extract parameters and call the calculate_sharpe_ratio tool immediately."),
                state["messages"][0]]
    return {"messages": [math_llm.invoke(messages)]}


def rag_retrieval_worker_node(state: TeamState):
    print("🔍 [RAG Context Worker] Querying local policy database...")
    user_query = state["messages"][0].content
    search_results = vector_store.similarity_search(user_query, k=1)
    state["metrics"].retrieved_mandate = search_results[0].page_content
    print(f"   [RAG HIT] Successfully matched text query to: {search_results[0].metadata.get('strategy', 'unknown')}")
    return {"metrics": state["metrics"]}


def validator_worker_node(state: TeamState):
    print("⚖️ [Risk Validator (Gemma 2)] Running parallel check against RAG mandate...")
    prompt = (
        f"Analyze if this Sharpe Ratio ({state['metrics'].sharpe_ratio}) matches the strategy mandate:\n"
        f"\"{state['metrics'].retrieved_mandate}\"\n"
        "End your analysis with exactly: 'OPTIMIZATION_REQUIRED: TRUE' or 'OPTIMIZATION_REQUIRED: FALSE'."
    )
    response = validator_llm.invoke([HumanMessage(content=prompt)])

    # Initialize a clean single-purpose update object to push to the reducer
    update = PortfolioMetrics()
    update.risk_validation_complete = True
    update.optimization_required = "OPTIMIZATION_REQUIRED: TRUE" in response.content.upper()
    return {"metrics": update}


def compliance_worker_node(state: TeamState):
    print("🔒 [Compliance Auditor (Mistral)] Running parallel regulatory tracking audit...")
    user_query = state["messages"][0].content
    prompt = (
        f"Review the user query details for strict compliance guidelines:\n\"{user_query}\"\n"
        "Check if the volatility levels look reasonable and safe. "
        "Conclude your analysis with exactly: 'COMPLIANCE_STATUS: PASSED' or 'COMPLIANCE_STATUS: FAILED'."
    )
    response = compliance_llm.invoke([HumanMessage(content=prompt)])

    # Initialize a clean single-purpose update object to push to the reducer
    update = PortfolioMetrics()
    update.compliance_audit_complete = True
    update.compliance_passed = "COMPLIANCE_STATUS: PASSED" in response.content.upper()
    return {"metrics": update}


def optimizer_worker_node(state: TeamState):
    print("🚀 [Optimizer Worker (Mistral)] Re-allocating portfolio allocations...")
    ratio = state["metrics"].sharpe_ratio
    response = optimizer_llm.invoke([HumanMessage(
        content=f"Portfolio sub-optimal at {ratio}. Execute optimize_portfolio_weights tool immediately.")])
    return {"messages": [response]}


def state_updater_node(state: TeamState):
    last_msg = state["messages"][-1]
    metrics = state.get("metrics", PortfolioMetrics())
    if getattr(last_msg, "type", "") == "tool" or last_msg.__class__.__name__ == "ToolMessage":
        content = str(last_msg.content)
        if "0." in content or "." in content and len(content) < 20:
            metrics.sharpe_ratio = round(float(content), 4)
        elif "Equities" in content:
            metrics.optimized_weights = {"allocations": content}
    return {"metrics": metrics}


# =====================================================================
# 5. GRAPH ASSEMBLY
# =====================================================================
builder = StateGraph(TeamState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("math_worker", math_worker_node)
builder.add_node("rag_retrieval_worker", rag_retrieval_worker_node)
builder.add_node("validator_worker", validator_worker_node)
builder.add_node("compliance_worker", compliance_worker_node)
builder.add_node("optimizer_worker", optimizer_worker_node)
builder.add_node("tools", tool_executor_node)
builder.add_node("state_updater", state_updater_node)

builder.set_entry_point("supervisor")

# Configure the conditional edge to execute list arrays
builder.add_conditional_edges("supervisor", lambda state: state["next_agents"], {
    "math_worker": "math_worker",
    "rag_retrieval_worker": "rag_retrieval_worker",
    "validator_worker": "validator_worker",
    "compliance_worker": "compliance_worker",
    "optimizer_worker": "optimizer_worker",
    "FINISH": END
})


def tool_routing_logic(state: TeamState):
    return "tools" if (
                hasattr(state["messages"][-1], "tool_calls") and state["messages"][-1].tool_calls) else "supervisor"


builder.add_conditional_edges("math_worker", tool_routing_logic, {"tools": "tools", "supervisor": "supervisor"})
builder.add_conditional_edges("optimizer_worker", tool_routing_logic, {"tools": "tools", "supervisor": "supervisor"})

builder.add_edge("tools", "state_updater")
builder.add_edge("state_updater", "supervisor")
builder.add_edge("rag_retrieval_worker", "supervisor")

# Merge parallel lanes back to supervisor sync point
builder.add_edge("validator_worker", "supervisor")
builder.add_edge("compliance_worker", "supervisor")

graph = builder.compile()

# =====================================================================
# 6. RUN THE PIPELINE
# =====================================================================
print("\n--- STARTING CHAPTER 6 PARALLEL FAN-OUT RUN ---")
query = "Check my client's portfolio performance under our Institutional Conservative Mandate rules. Return is 10%, volatility is 12%, and risk-free is 2%."

inputs = {
    "messages": [HumanMessage(content=query)],
    "next_agents": [],
    "metrics": PortfolioMetrics()
}

for output in graph.stream(inputs, stream_mode="updates"):
    for node, value in output.items():
        print(f"--> Done with Node: '{node}'")
