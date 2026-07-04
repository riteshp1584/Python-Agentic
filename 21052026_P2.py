import os
from dotenv import load_dotenv
from typing import TypedDict, Dict, Annotated, Any
from operator import add
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# Import our local Ollama model engine
from langchain_ollama import ChatOllama

load_dotenv()

class AdvancedQuantState(TypedDict):

    messages : Annotated[list[BaseMessage], add] # Slot 1: Conversational history tracker (appends via add reducer)

    portfolio_metrics : Dict[str, Any] # Slot 2: Dedicated isolated storage for structured financial metrics

# 2. INITIALIZE LOCAL LLAMA & DEFINE WORKFLOW NODES

llm = ChatOllama(
    model="llama3.1:latest",
    temperature=0.0,
    base_url="http://localhost:11434"
)

def extraction_node(state: AdvancedQuantState):
    """Desk 1: Extract structured parameters out of messy conversation text."""
    print("--- Node 1: Extracting raw text into portfolio_metrics storage slot ---")
    # Read the latest sentence from the chat history slot
    latest_user_message = state["messages"][-1].content

    extraction_prompt = (
        "You are a data extraction tool. Read the text below and extract the numerical values. "
        "Output ONLY a raw comma-separated line like this: return=X, risk_free=Y, volatility=Z\n"
        f"Text: {latest_user_message}"

    )

    response = llm.invoke(extraction_prompt)
    extracted_string = response.content.strip()

    print(f"[LOCAL LLM LOG] Extracted: {extracted_string}")

    # Update BOTH slots: Log a status in messages, and save the clean string in data storage
    return {
        "messages" : [AIMessage(content="[System Note: Extraction Complete.]")],
        "portfolio_metrics" : {"raw_data_string": extracted_string, "status": "EXTRACTED"}
    }

def calculation_node(State: AdvancedQuantState):
    """Desk 2: Use the data storage slot to summarize calculations without reading text."""
    print("--- Node 2: Accessing isolated portfolio_metrics storage slot ---")

    # Read straight from our dedicated storage slot
    saved_metrics = State["portfolio_metrics"]
    print(f"[LOCAL TOOL LOG] Calculation desk opened data bucket: {saved_metrics}")

    # Simulate processing the structured metrics cleanly
    final_report = f"Risk engine executed calculations successfully using: {saved_metrics['raw_data_string']}"

    return{
        "messages" : [AIMessage(content=final_report)]
    }

# 3. ASSEMBLE THE WORKFLOW LAYOUT

builder = StateGraph(AdvancedQuantState)

# Add our custom processing desks
builder.add_node("data_extractor", extraction_node)
builder.add_node("risk_calculator", calculation_node)

# Map the directional routing pipes
builder.add_edge(START, "data_extractor")
builder.add_edge("data_extractor", "risk_calculator")
builder.add_edge("risk_calculator", END)

graph =  builder.compile()

# 4. EXECUTION

print("\n--- STARTING LOCAL MULTI-VARIABLE STATE RUN ---")
query = "Portfolio Profile Analysis - Expected Return: 14.5%. Risk Free Rate: 3.5%. Portfolio Volatility: 16.0%."

inputs = {
    'messages' : [HumanMessage(content=query)],
    "portfolio_metrics" : {} # Initialize the secondary bucket as empty
}

# Stream the update transitions across your nodes
for output in graph.stream(inputs, stream_mode="updates"):
    for node, value in output.items():
        print(f"--> Finished Node: '{node}'\n")

