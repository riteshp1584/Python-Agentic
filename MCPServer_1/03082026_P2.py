import asyncio
import os
import sys
from dotenv import load_dotenv  # 1. Import load_dotenv

# 2. Load environment variables from the .env file in the root project folder
load_dotenv()

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

async def main():
    # 3. Read GOOGLE_API_KEY or GEMINI_API_KEY from environment
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set in your .env file!")

    # Configure stdio client to point to your server script
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["03082026_P1.py"]  # Path to your server script
    )

    print("Connecting to local FastMCP server via stdio...")

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Discover tools from server
            mcp_tools_response = await session.list_tools()
            print(f"Discovered {len(mcp_tools_response.tools)} tool(s).")

            # Map MCP tools to LangChain tools
            langchain_tools = []
            for tool in mcp_tools_response.tools:
                async def make_tool_call(tool_name=tool.name, **kwargs):
                    result = await session.call_tool(tool_name, kwargs)
                    return result.content

                langchain_tools.append(
                    Tool(
                        name=tool.name,
                        description=tool.description or "Local MCP Tool",
                        func=None,
                        coroutine=make_tool_call
                    )
                )

            # Initialize Gemini using loaded key
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0.0
            )

            # Create agent
            agent = create_react_agent(model=llm, tools=langchain_tools)

            # Execute query
            query = "What is the status of our database and analytics systems?"
            print(f"\nUser Query: '{query}'\n" + "=" * 50)

            response = await agent.ainvoke({"messages": [HumanMessage(content=query)]})

            # Print output
            final_message = response["messages"][-1]
            print("\nGemini Response:\n")
            print(final_message.content)

if __name__ == "__main__":
    asyncio.run(main())
