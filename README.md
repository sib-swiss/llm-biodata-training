# 🧑‍🏫 Using Large Language Models for Biodata Exploration

Course description: [sib.swiss/training/course/20260527_BAIBE](https://www.sib.swiss/training/course/20260527_BAIBE)

Pratical slides: [sib-swiss.github.io/llm-biodata-training](https://sib-swiss.github.io/llm-biodata-training)

- [Coding agents presentation](https://sib-swiss.github.io/llm-biodata-training/agents)
- [Tutorial app presentation](https://sib-swiss.github.io/llm-biodata-training/tutorial)

## 🚀 Deploy chat

Create `.env` file with providers API keys:

```sh
OPENROUTER_API_KEY=YYY
MISTRAL_API_KEY=YYY
OPENAI_API_KEY=sk-proj-YYY
```

Start the MCP server on http://localhost:8000

```sh
uv run mcp_server.py
```

In parallel start the chat webapp on http://localhost:8001

```sh
uv run chainlit run app.py --port 8001
```

Or start the 2 together on http://localhost:8000

```sh
uv run --env-file .env uvicorn main:app --port 8000 --workers 1
```

