## Building Agentic apps for Biodata Exploration

In this tutorial, you will build an LLM-powered bioinformatics assistant step by step using python.

---

## Outline

1. Call an LLM programmatically from Python
2. Connect to remote MCP servers
3. Build an agentic tool loop
4. Add a Chainlit web UI
5. Combine server & UI in a single app

---

## Summary

| Server | URL | What it does |
| --- | --- | --- |
| STRING-db | `https://mcp.string-db.org/` | Protein-protein interaction networks |
| Expasy | `https://chat.expasy.org/mcp/` | SPARQL queries over SIB databases (UniProt, OMA, Rhea, Bgee...) |
| New local | `http://localhost:8000` | Search UniProt and Rhea |

- StringDB questions
  - What are the top interaction partners of human TP53?
  - Find proteins functionally similar to BRCA1 in humans
  - What is the STRING network enrichment for TP53, BRCA1, and ATM?

- Expasy questions
  - What are the rat orthologs of human TP53? (OMA)
  - Find genes highly expressed in human liver (Bgee)
  - What biochemical reactions involve ATP hydrolysis? (Rhea)
  - Write a SPARQL query to find all human proteins involved in apoptosis (UniProt)
- New server: UniProt / Rhea
  - Search for reviewed human TP53 proteins in UniProt
  - What is the function of human BRCA1 and which diseases is it linked to?
  - What is the sequence of the enzyme that catalyzes ATP hydrolysis, and what reactions involve it?
  - Find the human enzyme for glucose phosphorylation and look up the Rhea reactions it catalyzes



---

## Part 1: Setup dependencies

Add more dependencies to the `pyproject.toml`:

```toml
[project]
name = "biodata-agent"
version = "0.0.1"
requires-python = "==3.13.*"
dependencies = [
    # [...]
    "chainlit >=2.8.1",
    "langchain >=1.3.1",
    "langchain-mcp-adapters >=0.2.2",
    "langchain-openai >=1.2.2",
    "langchain-ollama >=1.1.0",
    "langchain-mistralai >=1.1.4",
    "SQLAlchemy >=2.0.40",
]
```

---

## Part 1: LLM provider

First step is to choose your LLM provider, and get an API key

> - [OpenRouter](https://openrouter.ai/settings/keys) - hundreds of models from all providers, +5% on price per token compared to providers price, free tier
> - [MistralAI](https://console.mistral.ai/api-keys) - European provider, free tier
> - The usual suspects: Google, Anthropic, OpenAI...

If you don't have your own, we are providing an API key for OpenRouter model:

`openrouter/google/gemma-4-26b-a4b-it`

Create a `.env` file with the LLM provider API key(s):

```sh
OPENROUTER_API_KEY=YYY
```

---

## Part 1: Call an LLM from Python

Create an `app.py` file. Create a helper that picks the right LangChain class based on a `provider/model` string:

```python
import os
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

def load_chat_model(model: str) -> BaseChatModel:
    provider, model_name = model.split("/", maxsplit=1)
    if provider == "openrouter":
        return ChatOpenAI(
            model=model_name,
            base_url="https://openrouter.ai/api/v1",
            api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
            max_completion_tokens=2048,
        )
    if provider == "mistralai":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model_name=model_name, temperature=0, max_tokens=2048)
    raise ValueError(f"Unknown provider: {provider}")

llm = load_chat_model("openrouter/google/gemma-4-26b-a4b-it")
# llm = load_chat_model("mistralai/mistral-small-latest")
```

---

## Part 1: Invoke and stream

```python
async def main():
    question = "What are the rat orthologs of human TP53?"

    # Single call
    resp = llm.invoke(question)
    print(resp.content)

    # Streaming
    for chunk in llm.stream(question):
        print(chunk.content, end="", flush=True)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

Run with uv:

```sh
uv run --env-file .env app.py
```

---

## Part 1: Use a local LLM (optional)

Install [Ollama](https://ollama.com/download), pull a model (warning: ~4 GB):

```sh
ollama pull gemma4
ollama serve
```

Add to `load_chat_model`:

```python
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_name, temperature=0)

llm = load_chat_model("ollama/gemma4")
```

> Uses [llama.cpp](https://github.com/ggml-org/llama.cpp). Alternatively [vLLM](https://github.com/vllm-project/vllm) is better for serving multiple users.

---

## Part 2: Connect to a MCP server

The `mcp` package lets you connect directly to remote MCP servers over HTTP. Create `mcp_check.py`:

```python
import asyncio, json
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async def main() -> None:
    async with streamable_http_client("https://mcp.string-db.org/") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            print(f"Available tools ({len(tools_result.tools)}):")
            for tool in tools_result.tools:
                print(f"  - {tool.name}")
            # Call a specific tool:
            result = await session.call_tool(
                "string_all_interaction_partners",
                arguments={"identifiers": "TP53", "species": "9606", "required_score": "700", "limit": "5"},
            )
            print(json.dumps([c.model_dump() for c in result.content], indent=2))

asyncio.run(main())
```

Run it: `uv run mcp_check.py`

---

## Part 2: Connect to remote MCP servers

Add this to `app.py`. Use the `MultiServerMCPClient` connects to one or more MCP servers:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

async def get_mcp_tools():
    mcp_client = MultiServerMCPClient({
        "string": {
            "url": "https://mcp.string-db.org/",
            "transport": "streamable_http",
        },
        "expasy": {
            "url": "https://chat.expasy.org/mcp/",
            "transport": "streamable_http",
        },
    })
    return await mcp_client.get_tools()

async def main():
    tools = await get_mcp_tools()
    print(f"Found {len(tools)} tools:")
    for t in tools:
        print(f"  - {t.name}: {t.description[:80]}")
```

---

## Part 3: Build an agent

An **agent** is a model that calls tools in a loop until it has enough information to answer. Update `app.py`:

```python
from langchain.agents import create_agent

SYSTEM_PROMPT = """You are a bioinformatics assistant with access to SIB databases.
Always use tools to retrieve real data, never invent accessions or sequences.
For multi-step questions, chain tools: search -> get entry -> get interactions."""

async def main():
    tools = await get_mcp_tools()
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    question = "What are the top interaction partners of human TP53?"
    result = await agent.ainvoke({"messages": [("human", question)]})
    print(result["messages"][-1].content)
```

---

## Part 3: Stream agent steps

```python
from langchain_core.messages import AIMessageChunk, ToolMessage

async def main():
    tools = await get_mcp_tools()
    agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    question = "What are the interaction partners of TP53 and what reactions involve ATP?"

    async for chunk, __ in agent.astream(
        {"messages": [("human", question)]},
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            print(chunk.content, end="", flush=True)
        elif isinstance(chunk, ToolMessage):
            print(f"\n[Tool result: {str(chunk.content)[:200]}]\n")
```

> The agent decides which tools to call, in which order, how many times without you writing any routing logic.

---

## Part 4: Add a chat web UI

[Chainlit](https://docs.chainlit.io) is a Python framework that turns async functions into a chat UI. Replace the `main()` function approach with Chainlit lifecycle hooks.

```python
import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    """Called once when a new chat session opens."""
    tools = await get_mcp_tools()
    agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    cl.user_session.set("agent", agent)
    cl.user_session.set("chat_history", [])
```

Start the UI on <http://localhost:8000>:

```sh
# Codespace
chainlit run app.py

# Local with uv
uv run chainlit run app.py
```

---

## Part 4: Handle messages and stream tool steps

```python
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    chat_history: list = cl.user_session.get("chat_history") or []
    chat_history.append(HumanMessage(content=message.content))
    answer_msg = cl.Message(content="")
    pending_tool_calls: dict[str, dict] = {}
    current_tc_id: str | None = None
    async for chunk, _ in agent.astream({"messages": chat_history}, stream_mode="messages"):
        if isinstance(chunk, AIMessageChunk):
            if isinstance(chunk.content, str) and chunk.content:
                await answer_msg.stream_token(chunk.content)
            for tc_chunk in getattr(chunk, "tool_call_chunks", []) or []:
                tc_id = tc_chunk.get("id")
                if tc_id:
                    current_tc_id = tc_id
                    pending_tool_calls[tc_id] = {"name": tc_chunk.get("name", "tool"), "args": ""}
                elif current_tc_id:
                    pending_tool_calls[current_tc_id]["args"] += tc_chunk.get("args", "") or ""
        elif isinstance(chunk, ToolMessage):
            tc_id = getattr(chunk, "tool_call_id", "")
            info = pending_tool_calls.get(tc_id, {})
            async with cl.Step(name=f"Tool: {info.get('name', 'tool')}") as s:
                s.input = info.get("args", "")
                s.output = str(chunk.content)[:800]
            answer_msg = cl.Message(content="")

    if answer_msg.content:
        chat_history.append(AIMessage(content=answer_msg.content))
    cl.user_session.set("chat_history", chat_history[-20:])
    await answer_msg.send()
```

---

## Part 4: Starter questions and UI customization

```python
@cl.set_starters
async def set_starters(user: cl.User | None = None, language: str | None = None):
    return [
        cl.Starter(
            label="STRING interactions TP53",
            message="What are the interaction partners of TP53 with high confidence?",
        ),
        cl.Starter(
            label="Rat orthologs TP53",
            message="What are the rat orthologs of human TP53?",
        ),
        cl.Starter(
            label="ATP hydrolysis reactions",
            message="What biochemical reactions involve ATP hydrolysis in Rhea?",
        ),
    ]
```

Customize the UI in `.chainlit/config.toml`

---

## Part 5: Combine server & UI in one app

Uses the `mcp_server.py` built in the previous section. Create `main.py` to run the MCP server and Chainlit UI as a single FastAPI app without the need to start 2 separate processes:

```python
import contextlib
from collections.abc import AsyncIterator
from chainlit.utils import mount_chainlit
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from mcp_server import mcp

@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    async with mcp.session_manager.run():
        yield

app = FastAPI(title="Biodata Chat", lifespan=lifespan)
# MCP server on /mcp
app.mount("/mcp", mcp.streamable_http_app())
# Chainlit UI on /chat
mount_chainlit(app=app, target="app.py", path="/chat")

@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse(url="/chat")
```

---

## Part 5: Run the combined app

```sh
# Codespace
uvicorn main:app --host 0.0.0.0 --port 8000

# Local with uv
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

- Chainlit UI: <http://localhost:8000/chat>
- MCP endpoint: <http://localhost:8000/mcp>

> You can now point any MCP-compatible local client (Claude Code, codex, OpenCode) at `http://localhost:8000/mcp` to use your own tools.

---

## Part 5: ask questions

- Search for reviewed human TP53 proteins in UniProt
- What is the function of human BRCA1 and which diseases is it linked to?
- What is the sequence of the enzyme that catalyzes ATP hydrolysis, and what reactions involve it?
- Find the human enzyme for glucose phosphorylation and look up the Rhea reactions it catalyzes

---

## Advanced: Add auth and chat history persistence

> ⚠️ Does not work in GitHub Codespace because of port forwarding

Add password auth and SQLite-backed chat history to `app.py`:

```python
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> cl.User | None:
    if (username, password) == ("admin", "admin"):
        return cl.User(identifier="admin", metadata={"role": "ADMIN"})
    return None

@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo="sqlite+aiosqlite:///./data/chainlit.db")
```

Add to the `.env`: `CHAINLIT_AUTH_SECRET=your-secret-here`

Initialize the database before the first run:

```sh
uv run init_db.py
```

Then start the app as usual. Chainlit will now show a login screen and persist all conversations in `data/chainlit.db`.

---

## Advanced: implement the tool loop yourself

`create_agent` is a convenience wrapper. Under the hood it runs this loop:

```python
from langchain_core.messages import SystemMessage, ToolMessage

messages = [SystemMessage(content=SYSTEM_PROMPT), *chat_history]
llm_with_tools = llm.bind_tools(tools)
while True:
    response = await llm_with_tools.ainvoke(messages)
    messages.append(response)
    if not response.tool_calls:
        break  # no more tool calls, final answer ready
    for tool_call in response.tool_calls:
        tool = next(t for t in tools if t.name == tool_call["name"])
        result = await tool.ainvoke(tool_call["args"])
        messages.append(ToolMessage(
            content=str(result),
            tool_call_id=str(tool_call["id"]),
        ))
final_answer = response.content
```

> Each iteration: LLM decides which tools to call -> tools run -> results fed back -> LLM called again.

---

## Advanced: show tool steps in the Chainlit UI

Wrap each tool execution in a `cl.Step` to surface it in the chat:

```python
for tool_call in response.tool_calls:
    tool_name = tool_call["name"]
    matched_tool = next((t for t in tools if t.name == tool_name), None)
    async with cl.Step(name=f"Tool: {tool_name}") as step:
        step.input = str(tool_call["args"])
        if matched_tool is None:
            result = f"Error: tool '{tool_name}' not found"
        else:
            result = await matched_tool.ainvoke(tool_call["args"])
        step.output = str(result)[:800]  # truncate large payloads
    messages.append(ToolMessage(
        content=str(result),
        tool_call_id=str(tool_call["id"]),
    ))
```

Benefits of owning the loop:

- Add retry logic, timeouts, or per-tool auth headers
- Emit custom UI events between steps
- Log or trace every tool call independently

---

## References

| Resource | URL |
| --- | --- |
| UniProt | <https://uniprot.org> |
| Rhea DB | <https://www.rhea-db.org> |
| STRING MCP server | <https://mcp.string-db.org/> |
| Expasy MCP server | <https://chat.expasy.org/mcp/> |
| LangChain MCP adapters | <https://github.com/langchain-ai/langchain-mcp-adapters> |
| Chainlit docs | <https://docs.chainlit.io> |
| SIB Swiss Institute of Bioinformatics | https://www.sib.swiss |

---

## Thank you
