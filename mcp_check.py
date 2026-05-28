"""Check STRING-DB MCP server tool responses and their size."""

import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async def main() -> None:
    async with streamable_http_client("https://mcp.string-db.org/") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # List available tools
            tools_result = await session.list_tools()
            print(f"Available tools ({len(tools_result.tools)}):")
            for tool in tools_result.tools:
                print(f"  - {tool.name}")
            print()

            # Call string_all_interaction_partners for TP53
            result = await session.call_tool(
                "string_all_interaction_partners",
                arguments={"identifiers": "TP53", "species": "9606", "required_score": "700"},
            )
            raw = json.dumps([c.model_dump() for c in result.content])
            print(raw)


if __name__ == "__main__":
    asyncio.run(main())
