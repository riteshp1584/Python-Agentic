from fastmcp import FastMCP

mcp = FastMCP("LocalDataServer")


@mcp.tool(
    name="get_system_status",
    description="Fetch operational metrics for local system components ('database', 'analytics', 'cache')."
)
def get_system_status(service_name: str) -> str:
    statuses = {
        "database": "Operational - PostgreSQL active (Latency: 2ms)",
        "cache": "Operational - Redis active",
        "analytics": "Degraded - High memory utilization"
    }

    clean_input = str(service_name).lower().strip()

    # Search for matching keys in input
    matches = [f"[{k.upper()}]: {v}" for k, v in statuses.items() if k in clean_input]

    if matches:
        return "\n".join(matches)

    return f"Service '{service_name}' unknown. Available systems: {', '.join(statuses.keys())}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
