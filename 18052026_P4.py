import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from operator import add
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Using dotenv to fetch Google API Key from .env file
load_dotenv()

# 1. The Folder (State)
# We store a list of BaseMessage objects (Human or AI messages)
class GraphState(TypedDict):
    messages : Annotated[list[BaseMessage], add]

# 2. Initialize the Brain
# We'll use the fast, reliable gemini-2.5-flash model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

def assistant_node(state: GraphState):
    print("--- Assistant: AI is thinking... ---")

    # Read the entire chat history from the folder
    chat_history = state["messages"]

    # Hand the history to the LLM to get the next response
    response = llm.invoke(chat_history)

    # Return ONLY the new AI message. The 'add' reducer will append it automatically.
    return {"messages": [response]}

# 4. Building the Graph Layout
builder = StateGraph(GraphState)
builder.add_node("llm_assistant", assistant_node)
builder.set_entry_point("llm_assistant")
builder.add_edge("llm_assistant", END)

# Add the Checkpoint Filing Cabinet
memory_cabinet = MemorySaver()
graph = builder.compile(checkpointer=memory_cabinet)

# 5. Define our Session Passport (Thread ID)
config = {"configurable" : {"thread_id" : "quant_research_session_1"}}

# ROUND 1: Introduce yourself and test memory retention
print("\n--- ROUND 1 ---")
user_msg_1 = HumanMessage(content="Hi, my name is Ritesh. I am a quantitative research engineer.")
result_1 = graph.invoke({"messages" : [user_msg_1]}, config=config)

# Print the last message in the list (the AI's response)
print(f"AI Response: {result_1['messages'][-1].content}\n")

# ROUND 2: Test if it remembers your name and background from the filing cabinet
print("\n--- ROUND 2 ---")
user_msg_2 = HumanMessage(content="What was my name again, and what field do I work in?")
result_2 = graph.invoke({"messages" : [user_msg_2]}, config=config)

print(f"AI Response: {result_2['messages'][-1].content}\n")
