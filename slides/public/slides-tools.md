## Using agentic tools for Biodata Exploration

In this presentation, we will introduce agentic tools, and how to use them with little code.

---

## Outline

1. Use remote MCP servers in a web chat UI
2. Use remote MCP servers in a desktop agent
3. Use MCP server with authentication 
4. Use MCP server through stdio (local)

---

## Questions

Related to proteins interactions:

- What are the top interaction partners of human TP53?
- Find proteins functionally similar to BRCA1 in humans
- What is the STRING network enrichment for TP53, BRCA1, and ATM?

Related to SIB databases:

- What are the rat orthologs of human TP53? (OMA)
- Find genes highly expressed in human liver (Bgee)
- What biochemical reactions involve ATP hydrolysis? (Rhea)
- Write a SPARQL query to find all human proteins involved in apoptosis (UniProt)

---

## Searching for solutions

Official registry effort: [registry.modelcontextprotocol.io](https://registry.modelcontextprotocol.io)

GitHub curated MCP registry: [github.com/mcp](https://github.com/mcp)

Biodata-focused registry: [biocontext.ai/registry](https://biocontext.ai/registry)

Skills registry: [skills.sh](https://www.skills.sh/)

---

## Use MCP servers in a web chat UI

No code required just point the chat app at a remote MCP server URL.

MCP servers we will use:

| Server | URL | What it does |
| --- | --- | --- |
| STRING-db | `https://mcp.string-db.org/` | Protein-protein interaction networks |
| Expasy | `https://chat.expasy.org/mcp/` | SPARQL queries over SIB databases (UniProt, OMA, Rhea, Bgee...) |

Both are public, free, no authentication required.

---

## Add STRING-db to Mistral.ai Chat

1. Go to [chat.mistral.ai](https://chat.mistral.ai)
2. Click **Agents** in the sidebar > **Connectors** > **Add Connector** > **Custom MCP Connector**
3. Server address: `https://mcp.string-db.org/`
4. Le Chat will discover the available tools automatically
5. Click **+** > **Connectors** to see your custom connectors

Try these questions:

- What are the top interaction partners of human TP53?
- Find proteins functionally similar to BRCA1 in humans
- What is the STRING network enrichment for TP53, BRCA1, and ATM?

---

## Add Expasy to ChatGPT

1. Open [chatgpt.com](https://chatgpt.com) (desktop app or web)
2. Go to **Apps** in the sidebar > ⚙️ top right > **Create app**
3. Server URL: `https://chat.expasy.org/mcp/`
4. ChatGPT will use the tools from the Expasy server to help write SPARQL queries against SIB endpoints

Try these questions:

- What are the rat orthologs of human TP53? (OMA)
- Find genes highly expressed in human liver (Bgee)
- What biochemical reactions involve ATP hydrolysis? (Rhea)
- Write a SPARQL query to find all human proteins involved in apoptosis (UniProt)

---

## What just happened?

```txt
Web UI  -->  (discovers tools)  -->  MCP server (string-db or expasy)
              <tool list>

User asks question
  --> LLM decides which tool to call
  --> Web UI calls MCP server tool
  --> Result returned to LLM
  --> LLM synthesizes the answer
```

This is the **agentic loop**: the LLM uses tools in a loop to answer complex questions.

---

## Connect a MCP server to a coding agent

Use your favorite coding agent (Claude Code, Codex, Cursor, OpenCode, GitHub Copilot)

For [OpenCode](https://opencode.ai/download), open the user config file `~/.config/opencode/opencode.jsonc`

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "expasy-mcp": {
      "type": "remote",
      "url": "https://chat.expasy.org/mcp/",
      "enabled": true
    },
    "stringdb": {
      "type": "remote",
      "url": "https://mcp.string-db.org/",
      "enabled": true
    }
  },
  "provider": {
    "cesnet": {
      "name": "Cesnet",
      "npm": "@ai-sdk/openai-compatible",
      "models": {
        "qwen3-coder": {"name": "qwen3-coder"}
      },
      "options": {
        "baseURL": "https://llm.ai.e-infra.cz/v1"
      }
    }
  }
```

---

## MCP with authentication 

HuggingFace MCP server uses an API key to pass as env variable

Instructions for various LLM clients here: [huggingface.co/mcp?login](https://huggingface.co/mcp?login)

For OpenCode:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "huggingface": {
      "type": "remote",
      "url": "https://huggingface.co/mcp",
      "headers": {
        "Authorization": "Bearer <HF_TOKEN>"
      },
      "enabled": true
    }
  }
}
```

Example question: Show datasets about weather time-series available on huggingface

---

## MCP with stdio

Local MCP servers run as a subprocess via `stdio` - requires the tool installed locally

Requires [`uv`](https://docs.astral.sh/uv) installed

For OpenCode:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "pubmed": {
      "type": "local",
      "command": ["uvx", "pubmedmcp"],
      "enabled": true
    }
  }
}
```

Example questions:

- What are recent clinical trials using mRNA vaccines for cancer treatment?
- Find papers about AlphaFold and its applications in drug discovery

---

## Use skills

Requires a place to execute the code, e.g. your laptop

Find skill for UniProt: [skills.sh](https://www.skills.sh)

---

## References

| Resource | URL |
| --- | --- |
| UniProt | <https://uniprot.org> |
| Rhea DB | <https://www.rhea-db.org> |
| STRING MCP server | <https://mcp.string-db.org/> |
| Expasy MCP server | <https://chat.expasy.org/mcp/> |
| SIB Swiss Institute of Bioinformatics | https://www.sib.swiss |

---

## Thank you
