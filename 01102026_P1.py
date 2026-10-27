import os
from typing import TypedDict, Optional, Dict
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# =====================================================================
# 1. STATE DEFINITIONS & SCOPES
# =====================================================================

# --- PARENT STATE (The Executive Ledger) ---
class ExecutiveState(TypedDict):
    client_id: str
    portfolio_type: str  # e.g., "Conservative Mandate"
    final_allocations: Optional[Dict[str, str]] = None
    executive_summary: Optional[str] = None
    approval_status: Optional[str] = None  # "APPROVED" or "REJECTED"


# --- CHILD STATE (The Sandbox Scratchpad) ---
class LocalSubMetrics(BaseModel):
    sharpe_ratio: Optional[float] = None
    risk_validation_complete: bool = False
    optimization_required: Optional[bool] = None


class AnalystSandboxState(TypedDict):
    sandbox_messages: list[str]  # Private short-term thread (Saves RAM!)
    metrics: LocalSubMetrics  # Private local database metrics


# =====================================================================
# 2. BOUNDARY MAPPING GATEWAYS (THE ENTRY & EXIT PORTS)
# =====================================================================

def parent_to_child_entry(parent_state: ExecutiveState) -> dict:
    """Executed automatically when entering the child sub-graph."""
    print("\n   📥 [BOUNDARY ENTRY] Instantiating isolated child sandbox state...")
    print(f"      -> Copying 'portfolio_type' ({parent_state['portfolio_type']}) down to sub-graph.")

    return {
        "sandbox_messages": [f"Target Mandate Profile: {parent_state['portfolio_type']}"],
        "metrics": LocalSubMetrics()  # Fresh, blank local database state for the child
    }


def child_to_parent_exit(child_state: AnalystSandboxState) -> dict:
    """Executed automatically when exiting the child sub-graph."""
    print("\n   📤 [BOUNDARY EXIT] Extracting final verdict and offloading sandbox from RAM...")

    child_metrics = child_state["metrics"]
    is_failed = child_metrics.optimization_required

    verdict = "REJECTED_NEEDS_REBALANCING" if is_failed else "APPROVED"
    print(f"      -> Porting verdict ({verdict}) up to Parent Ledger.")

    return {
        "approval_status": verdict
    }


# =====================================================================
# 3. CHILD SUB-GRAPH WORKERS & ASSEMBLY
# =====================================================================

def child_math_worker(state: AnalystSandboxState):
    print("      🔬 [Child Team -> Math Node] Calculating Sharpe Ratio locally...")

    updated_metrics = state["metrics"].model_copy()
    updated_metrics.sharpe_ratio = 0.52

    return {
        "sandbox_messages": state["sandbox_messages"] + ["Calculated Sharpe ratio successfully."],
        "metrics": updated_metrics
    }


def child_validator_worker(state: AnalystSandboxState):
    print("      ⚖️ [Child Team -> Risk Node] Checking calculated ratio against rules...")

    updated_metrics = state["metrics"].model_copy()

    if updated_metrics.sharpe_ratio is not None and updated_metrics.sharpe_ratio < 0.70:
        updated_metrics.optimization_required = True
    else:
        updated_metrics.optimization_required = False

    updated_metrics.risk_validation_complete = True

    return {
        "sandbox_messages": state["sandbox_messages"] + ["Risk validation pipeline complete."],
        "metrics": updated_metrics
    }


# Assemble Independent Child Graph Sandbox
child_builder = StateGraph(AnalystSandboxState)
child_builder.add_node("inner_math", child_math_worker)
child_builder.add_node("inner_validator", child_validator_worker)

child_builder.add_edge(START, "inner_math")
child_builder.add_edge("inner_math", "inner_validator")
child_builder.add_edge("inner_validator", END)

compiled_child_graph = child_builder.compile()


# =====================================================================
# 4. PARENT GRAPH NODES, ROUTERS & ASSEMBLY
# =====================================================================

def executive_supervisor(state: ExecutiveState):
    print("\n👑 [Parent Graph -> Executive Supervisor] Assessing milestone checkpoints...")

    if state.get("approval_status") is None:
        print("   -> Current State: No analysis metrics found.")
        print("   -> Action Decision: Route to Analytics Sub-Graph Team.")
    else:
        print(f"   -> Current State: Analysis complete with status: {state['approval_status']}.")
        print("   -> Action Decision: Route to Final Reporting Engine.")

    return {}


def executive_routing_logic(state: ExecutiveState) -> str:
    """Pure routing function for the conditional edge choice."""
    if state.get("approval_status") is None:
        return "run_analytics_team"
    return "finalize_report"


def run_analytics_team_gateway(state: ExecutiveState):
    """Explicit Gateway Node: Orchestrates state transition safely."""
    child_inputs = parent_to_child_entry(state)

    print("   🚀 [Gateway] Launching isolated Child Sub-Graph Engine...")
    child_outputs = compiled_child_graph.invoke(child_inputs)
    print("   🏁 [Gateway] Child Sub-Graph Engine execution finished.")

    parent_updates = child_to_parent_exit(child_outputs)
    return parent_updates


def final_report_node(state: ExecutiveState):
    print("\n📝 [Parent Graph -> Reporting Engine] Packaging final client artifact...")

    status = state["approval_status"]
    client = state["client_id"]

    summary = f"=== FINAL PORTFOLIO STATUS FOR CLIENT: {client} ===\n"
    summary += f"Status Verdict: {status}\n"
    if status == "APPROVED":
        summary += "Action Plan: Portfolio balances conform to mandate guidelines. No changes made."
    else:
        summary += "Action Plan: Trigger trade executions immediately to buffer risk exposures."

    return {"executive_summary": summary}


# Assemble Main Parent System Graph
parent_builder = StateGraph(ExecutiveState)

parent_builder.add_node("executive_supervisor", executive_supervisor)
parent_builder.add_node("run_analytics_team", run_analytics_team_gateway)
parent_builder.add_node("final_report", final_report_node)

# Set Graph Routing Pathways
parent_builder.set_entry_point("executive_supervisor")

parent_builder.add_conditional_edges(
    "executive_supervisor",
    executive_routing_logic,
    {
        "run_analytics_team": "run_analytics_team",
        "finalize_report": "final_report"
    }
)

parent_builder.add_edge("run_analytics_team", "executive_supervisor")
parent_builder.add_edge("final_report", END)

# --- LEVEL 5 INTEGRATION: CHECKPOINTER & INTERRUPT GATE ---
memory_checkpointer = MemorySaver()

compiled_parent_system = parent_builder.compile(
    checkpointer=memory_checkpointer,
    interrupt_before=["final_report"]  # Interrupt gate before report generation
)
print("🏗️ Hierarchical System Architecture compiled with HITL Checkpointer.")

# =====================================================================
# 5. EXECUTION PIPELINE RUNNER (HITL INCLUDED)
# =====================================================================
if __name__ == "__main__":
    print("\n--- STARTING LEVEL 5 HITL COGNITIVE RUN ---")

    # Session Thread Configuration
    thread_config = {"configurable": {"thread_id": "CLIENT_BLR_9921_SESSION"}}

    initial_input = {
        "client_id": "BLR-QUANT-9921",
        "portfolio_type": "Institutional Conservative Mandate",
        "final_allocations": None,
        "executive_summary": None,
        "approval_status": None
    }

    # -----------------------------------------------------------------
    # PHASE 1: Autonomous Execution until Interrupt Gate
    # -----------------------------------------------------------------
    print("\n⚡ [PHASE 1] Executing Graph Autonomous Nodes...")
    for output in compiled_parent_system.stream(initial_input, thread_config, stream_mode="updates"):
        for node_name, value in output.items():
            print(f"📡 [System Broadcast] Node Event: '{node_name}' completed.")

    # -----------------------------------------------------------------
    # PHASE 2: Human Inspection & Decision Prompt
    # -----------------------------------------------------------------
    snapshot = compiled_parent_system.get_state(thread_config)
    print("\n" + "=" * 60)
    print("⏸️ [HUMAN-IN-THE-LOOP INTERRUPT TRIGGERED]")
    print(f"   Next Pending Node : {snapshot.next}")
    print(f"   Current Verdict   : {snapshot.values.get('approval_status')}")
    print("=" * 60)

    print("\nOptions:")
    print(" [1] Approve & Proceed to Report Generation")
    print(" [2] Override State (Change verdict to 'APPROVED') & Proceed")
    print(" [3] Abort Execution")

    choice = input("\nSelect Action [1/2/3]: ").strip()

    if choice == "2":
        print("\n✏️ [HUMAN OVERRIDE] Injecting state update: approval_status = 'APPROVED'")
        compiled_parent_system.update_state(
            thread_config,
            {"approval_status": "APPROVED"}
        )
        snapshot = compiled_parent_system.get_state(thread_config)
        print(f"   Updated Verdict in Checkpoint: {snapshot.values.get('approval_status')}")

    if choice in ["1", "2"]:
        # -------------------------------------------------------------
        # PHASE 3: Resuming Execution from Checkpoint Snapshot
        # -------------------------------------------------------------
        print("\n▶️ [PHASE 2] Resuming Graph Execution from Checkpoint...")
        for output in compiled_parent_system.stream(None, thread_config, stream_mode="updates"):
            for node_name, value in output.items():
                print(f"📡 [System Broadcast] Node Event: '{node_name}' completed.")
                if isinstance(value, dict) and "executive_summary" in value and value["executive_summary"]:
                    print(f"\n{value['executive_summary']}\n")
    else:
        print("\n❌ Execution halted by human supervisor. Thread saved at checkpoint.")
