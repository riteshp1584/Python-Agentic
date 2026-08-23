
from fastmcp import FastMCP

mcp = FastMCP("LocalDataServer")


@mcp.tool(
    name="get_system_status",
    description="Fetch operational metrics for a SINGLE system component. Call this function once per component (e.g. call once for 'database', once for 'analytics')."
)
def get_system_status(service_name: str) -> str:
    """Fetch metrics for a given service.

    Args:
        service_name: A single service identifier, such as 'database', 'analytics', or 'cache'.
    """
    statuses = {
        "database": "Operational - PostgreSQL active (Latency: 2ms)",
        "cache": "Operational - Redis active",
        "analytics": "Degraded - High memory utilization"
    }

    clean_input = service_name.lower().strip()

    # Direct match
    if clean_input in statuses:
        return f"[{clean_input.upper()}]: {statuses[clean_input]}"

    # Fallback multi-key check if LLM passes combined strings like "database and analytics"
    found_matches = []
    for key, val in statuses.items():
        if key in clean_input:
            found_matches.append(f"[{key.upper()}]: {val}")

    if found_matches:
        return "\n".join(found_matches)

    return f"Service '{service_name}' unknown. Available systems: {', '.join(statuses.keys())}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
