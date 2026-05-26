"""Chat app with an agentic tool-call loop over MCP servers tools."""

import os

import chainlit as cl

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

LLM_MODEL = "cesnet/qwen3-coder"
# LLM_MODEL = "ollama/gemma4"
# LLM_MODEL = "mistralai/mistral-small-latest"

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


@cl.on_chat_start
async def on_chat_start():
    mcp_client = MultiServerMCPClient({
        "biodata": {
            "url": MCP_SERVER_URL,
            "transport": "streamable_http",
        },
        "string": {
            "url": "https://mcp.string-db.org/",
            "transport": "streamable_http",
        },
    })
    tools = await mcp_client.get_tools()
    llm = load_chat_model(LLM_MODEL)
    cl.user_session.set("tools", tools)
    cl.user_session.set("llm", llm.bind_tools(tools))
    cl.user_session.set("chat_history", [])


@cl.on_message
async def on_message(message: cl.Message):
    """Handle each user message through a manual agentic tool-calling loop."""
    tools: list = cl.user_session.get("tools") or []
    llm_with_tools = cl.user_session.get("llm")
    chat_history: list = cl.user_session.get("chat_history") or []

    if llm_with_tools is None:
        await cl.Message(content="Agent not initialized.").send()
        return

    chat_history.append(HumanMessage(content=message.content))
    # Build the message list the LLM sees: system prompt + full conversation
    messages: list = [SystemMessage(content=SYSTEM_PROMPT), *chat_history]

    answer_msg = cl.Message(content="")

    # Tool-calling loop: keep invoking the LLM until it stops requesting tools
    while True:
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            # No tool calls in this response - it is the final answer
            break

        # Some models emit reasoning text alongside tool calls; show it
        if response.content and isinstance(response.content, str):
            await answer_msg.stream_token(response.content)

        # Execute each requested tool and feed the result back into messages
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = str(tool_call.get("id") or tool_name)
            matched_tool = next((t for t in tools if t.name == tool_name), None)
            async with cl.Step(name=f"Tool: {tool_name}") as step:
                step.input = str(tool_args)
                if matched_tool is None:
                    result = f"Error: tool '{tool_name}' not found"
                else:
                    result = await matched_tool.ainvoke(tool_args)
                step.output = str(result)[:800]

            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))
        # Reset the answer message for the next LLM reply
        answer_msg = cl.Message(content="")

    # Stream the final answer token by token
    final_content = response.content if isinstance(response.content, str) else ""
    for token in final_content:
        await answer_msg.stream_token(token)

    if final_content:
        chat_history.append(AIMessage(content=final_content))
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


@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo="sqlite+aiosqlite:///./data/chainlit.db")
