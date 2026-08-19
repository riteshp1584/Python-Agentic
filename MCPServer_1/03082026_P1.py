# mcp_server.py
from fastmcp import FastMCP

# Initialize local FastMCP Server
mcp = FastMCP("LocalDataServer")

@mcp.tool()
def get_system_status(service_name: str) -> str:
    """Fetch operational metrics for local infrastructure services."""
    statuses = {
        "database": "Operational - PostgreSQL 16 active (Latency: 2ms)",
        "cache": "Operational - Redis cluster active",
        "analytics": "Degraded - Memory consumption at 88%"
    }
    return statuses.get(service_name.lower(), f"Service '{service_name}' status unknown.")

if __name__ == "__main__":
    # Runs standard I/O communication over local stdin/stdout process
    mcp.run(transport="stdio")
