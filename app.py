"""Chat app with an agentic tool-call loop over MCP servers tools."""

import os

import chainlit as cl

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

LLM_MODEL="cesnet/qwen3-coder"
# LLM_MODEL="ollama/gemma4"
# LLM_MODEL="mistralai/mistral-small-latest"

SYSTEM_PROMPT = """You are a bioinformatics assistant with access to SIB databases.
Always use tools to retrieve real data, never invent accessions or sequences.
For multi-step questions, chain tools: search -> get entry -> get interactions.
"""

MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"

def load_chat_model(model: str) -> BaseChatModel:
    provider, model_name = model.split("/", maxsplit=1)
    if provider == "cesnet":
        return ChatOpenAI(
            model=model_name,
            base_url="https://llm.ai.e-infra.cz/v1",
            api_key=SecretStr(os.environ["CESNET_API_KEY"]),
            max_completion_tokens=2048,
        )
    if provider == "openrouter":
        return ChatOpenAI(
            model=model_name,
            base_url="https://openrouter.ai/api/v1",
            api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
            max_completion_tokens=2048,
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_name, temperature=0)
    if provider == "mistralai":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model_name=model_name, temperature=0, max_tokens=2048)
    raise ValueError(f"Unknown provider: {provider}")



async def init_agent():
    mcp_client = MultiServerMCPClient({
        "biodata": {
            "url": MCP_SERVER_URL,
            "transport": "streamable_http",
        },
        "string": {
            "url": "https://mcp.string-db.org/",
            "transport": "streamable_http",
        },
        "expasy": {
            "url": "https://chat.expasy.org/mcp/",
            "transport": "streamable_http",
        },
    })
    tools = await mcp_client.get_tools()
    llm = load_chat_model(LLM_MODEL)
    return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


@cl.on_chat_start
async def on_chat_start():
    agent = await init_agent()
    cl.user_session.set("agent", agent)
    cl.user_session.set("chat_history", [])


@cl.on_message
async def on_message(message: cl.Message):
    """Handle each user message through the agentic tool loop."""
    agent = cl.user_session.get("agent")
    chat_history: list = cl.user_session.get("chat_history") or []
    if agent is None:
        await cl.Message(content="Agent not initialized.").send()
        return

    chat_history.append(HumanMessage(content=message.content))
    answer_msg = cl.Message(content="")
    pending_tool_calls: dict[str, dict] = {}
    current_tc_id: str | None = None

    async for chunk, __ in agent.astream(
        {"messages": chat_history}, stream_mode="messages"
    ):
        if isinstance(chunk, AIMessageChunk):
            if isinstance(chunk.content, str) and chunk.content:
                await answer_msg.stream_token(chunk.content)
            for tc_chunk in getattr(chunk, "tool_call_chunks", []) or []:
                tc_id = tc_chunk.get("id")
                if tc_id:
                    current_tc_id = tc_id
                    pending_tool_calls[tc_id] = {
                        "name": tc_chunk.get("name", "tool"),
                        "args": tc_chunk.get("args", "") or "",
                    }
                elif current_tc_id:
                    pending_tool_calls[current_tc_id]["args"] += tc_chunk.get("args", "") or ""
        elif isinstance(chunk, ToolMessage):
            tc_id = getattr(chunk, "tool_call_id", "")
            info = pending_tool_calls.get(tc_id, {})
            async with cl.Step(name=f"🛠 {info.get('name', 'tool')}") as s:
                s.input = info.get("args", "")
                s.output = str(chunk.content)[:800]
            answer_msg = cl.Message(content="")

    final_answer = answer_msg.content
    if final_answer:
        chat_history.append(AIMessage(content=final_answer))
    cl.user_session.set("chat_history", chat_history[-20:])
    await answer_msg.send()


@cl.set_starters
async def set_starters(user: cl.User | None = None, language: str | None = None):
    return [
        cl.Starter(
            label="UniProt disease variants BRCA1",
            message="What is the function of human BRCA1 and which diseases is it linked to?",
        ),
        cl.Starter(
            label="Rhea reactions ATP hydrolysis",
            message="Find reactions involving ATP hydrolysis in Rhea",
        ),
        cl.Starter(
            label="STRING interactions TP53",
            message="What are the interaction partners of TP53 with high confidence?",
        ),
    ]

@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> cl.User | None:
    if (username, password) == ("admin", "admin"):
        return cl.User(identifier="admin", metadata={"role": "ADMIN"})
    return None

# Enable persistence with SQLite
@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo="sqlite+aiosqlite:///./data/chainlit.db")
