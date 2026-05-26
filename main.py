"""Combine the MCP server and Chainlit UI in a single FastAPI app."""

import contextlib
from collections.abc import AsyncIterator

from chainlit.utils import mount_chainlit
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from mcp_server import mcp

@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan that initializes the MCP session manager."""
    try:
        async with mcp.session_manager.run():
            yield
    finally:
        pass

app = FastAPI(
    title="Biodata chat",
    description="Chat app and MCP tools for Biodata exploration.",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount MCP server at /mcp
app.mount("/mcp", mcp.streamable_http_app())
# Mount Chainlit UI at /chat
mount_chainlit(app=app, target="app.py", path="/chat")

@app.get("/")
async def redirect_to_chat() -> RedirectResponse:
    return RedirectResponse(url="/chat")
