import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

async def main():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set!")

    env = os.environ.copy()
    env["FASTMCP_SHOW_BANNER"] = "false"

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["03082026_P5.py"],
        env=env
    )

    print("Connecting to local FastMCP server via stdio...")

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()
            print(f"Discovered {len(mcp_tools.tools)} tool(s).")

            langchain_tools = []
            for tool in mcp_tools.tools:
                def make_executor(tool_name):
                    async def _call(**kwargs):
                        res = await session.call_tool(tool_name, kwargs)
                        if hasattr(res, 'content') and isinstance(res.content, list):
                            texts = [item.text for item in res.content if hasattr(item, 'text')]
                            return "\n".join(texts)
                        return str(res)
                    return _call

                langchain_tools.append(
                    Tool(
                        name=tool.name,
                        description=tool.description or "Local MCP Tool",
                        func=None,
                        coroutine=make_executor(tool.name)
                    )
                )

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0.0
            )

            agent = create_agent(model=llm, tools=langchain_tools)

            query = "What is the status of our database and analytics systems?"
            print(f"\nUser Query: '{query}'\n" + "=" * 50)

            response = await agent.ainvoke({"messages": [HumanMessage(content=query)]})

            final_message = response["messages"][-1]
            print("\nGemini Response:\n")
            print(final_message.content)

if __name__ == "__main__":
    asyncio.run(main())
