import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Dict, Any, Literal
from operator import add
from pydantic import BaseModel, Field # <-- NEW: Enforces structural compliance
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama

load_dotenv()

# 1. THE STATE DEFINITION

class TeamState (TypedDict):
    messages: Annotated[list[BaseMessage], add]
    next_agent : str

# 2. THE MATHEMATICAL TOOL

def calculate_sharpe_ratio(portfolio_return : float, risk_free_rate : float, standard_deviation : float) -> float:
    print(f"\n⚡ [SYSTEM MATH ENGINE] Firing local Python function...")
    return (portfolio_return - risk_free_rate) / standard_deviation

sharpe_tool_object = StructuredTool.from_function(
    func=calculate_sharpe_ratio,
    name="calculate_sharpe_ratio",
    description="Calculates the Sharpe Ratio of an investment portfolio to evaluate risk-adjusted performance."
)

tools_list = [sharpe_tool_object]
tool_executor_node = ToolNode(tools_list)

# 3. ROUTING SCHEMA & INITIALIZE MODELS
# We define a rigid Pydantic object that Llama 3.1 MUST fill out

class RouterSchema(BaseModel):
    """Decide which desk should process the portfolio file next."""
    next_agent: Literal["math_worker", "validator_worker", "FINISH"] = Field(description="Choose 'math_worker' for calculations, 'validator_worker' for commentary/review, or 'FINISH' if both tasks are fully done.")

raw_llm = ChatOllama(
    model="llama3.1:latest",
    temperature=0.0,
    base_url="http://localhost:11434",
    keep_alive='5s'
)

# We compile the supervisor to strictly produce our structural RouterSchema format
supervisor_llm = raw_llm.with_structured_output(RouterSchema)

math_llm = ChatOllama(
    model="mistral:latest",
    temperature=0.0,
    base_url="http://localhost:11434",
    keep_alive='5s'
).bind_tools(tools_list)

validator_llm = ChatOllama(
    model="gemma2:2b",
    temperature=0.0,
    base_url="http://localhost:11434",
    keep_alive='5s'
)

# 4. NODES

def supervisor_node (state: TeamState):
    print("\n👑 [Supervisor (Llama 3.1)] Checking history and forcing routing decision...")

    prompt = (
        "You are the Portfolio Supervisor Manager. Look at the execution logs history. "
        "Your team must follow this strict sequential chain of command:\n"
        "1. If the 'calculate_sharpe_ratio' tool has NOT run yet, you MUST choose 'math_worker'.\n"
        "2. If the tool HAS run and returned a number, but 'Validator Final Review' is NOT in the logs, you MUST choose 'validator_worker'.\n"
        "3. Only choose 'FINISH' if both a tool result and a 'Validator Final Review' are explicitly present."
    )

    messages = [HumanMessage(content=prompt)] + state["messages"]
    # The output is guaranteed to be a structured RouterSchema object
    response : RouterSchema = supervisor_llm.invoke(messages)

    print(f"   >> Forced Supervisor Output Struct: next_agent='{response.next_agent}'")
    return {'next_agent': response.next_agent}

def math_worker_node (state: TeamState):
    print("🔬 [Math Worker (Mistral)] Processing numbers...")
    worker_prompt = (
        "You are a strict math tool call assistant. Your ONLY job is to call the calculate_sharpe_ratio tool. "
        "Do not output preambles, summaries, or conversational sentences in plain text. Just call the tool."
    )
    messages = [HumanMessage(content=worker_prompt)] + state["messages"]
    response = math_llm.invoke(messages)
    return {"messages": [response]}

def validator_worker_node (state: TeamState):
    print("⚖️ [Validator (Gemma 2)] Inspecting math outputs...")
    prompt = (
        "Look at the actual tool return value (0.6111) in the history. Provide a short "
        "one-sentence financial evaluation explaining whether this specific Sharpe Ratio is healthy. "
        "Start your response explicitly with the words 'Validator Final Review:'."
    )
    messages = state["messages"] + [HumanMessage(content=prompt)]
    response = validator_llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)]}

# 5. GRAPH ASSEMBLY

builder = StateGraph(TeamState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("math_worker", math_worker_node)
builder.add_node("tools", tool_executor_node)
builder.add_node("validator_worker", validator_worker_node)

builder.set_entry_point("supervisor")

# Conditional Router for the Supervisor
def supervisor_router (state: TeamState):
    if state["next_agent"] == "math_worker":
        return "math_worker"
    elif state["next_agent"] == "validator_worker":
        return "validator_worker"
    else:
        return "__end__"

builder.add_conditional_edges("supervisor", supervisor_router, {
    "math_worker" : "math_worker",
    "validator_worker" : "validator_worker",
    "__end__" : END
})

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
        print(f"--> Done with Node: '{node}'")
        if "messages" in value and value["messages"]:
            print(f"    Log Content: {value['messages'][-1].content}\n")

