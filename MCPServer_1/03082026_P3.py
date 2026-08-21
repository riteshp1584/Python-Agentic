from fastmcp import FastMCP

mcp = FastMCP("LocalDataServer")

@mcp.tool()
def get_system_status(service_name: str) -> str:
    """Fetch operational metrics for a specific infrastructure service.

    Args:
        service_name: Name of a single service, e.g., 'database', 'analytics', or 'cache'.
    """
    query = service_name.lower().strip()

    statuses = {
        "database": "Operational - PostgreSQL active (Latency: 2ms)",
        "cache": "Operational - Redis active",
        "analytics": "Degraded - High memory utilization"
    }

    # Match exact or partial keys if LLM combines terms
    matched_results = []
    for key, status in statuses.items():
        if key in query:
            matched_results.append(f"{key.capitalize()}: {status}")

    if matched_results:
        return "\n".join(matched_results)

    return f"Service '{service_name}' status unknown. Available services: {', '.join(statuses.keys())}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
