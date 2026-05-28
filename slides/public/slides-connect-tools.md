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

Related to proteins interactions ([STRING-db](https://string-db.org)):

- What are the top interaction partners of human TP53?
- Find proteins functionally similar to BRCA1 in humans
- What is the STRING network enrichment for TP53, BRCA1, and ATM?

Related to SIB biological databases ([expasy portal](https://www.expasy.org/chat)):

- What are the rat orthologs of human TP53? ([OMA](https://omabrowser.org/oma/home/) orthology)
- Find genes highly expressed in human liver ([Bgee](https://www.bgee.org) gene expression)
- What biochemical reactions involve ATP hydrolysis? ([Rhea](https://www.rhea-db.org) chemical reactions)
- Write a SPARQL query to find all human proteins involved in apoptosis ([UniProt](https://www.uniprot.org) proteins)

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
| ExpasyGPT | `https://chat.expasy.org/mcp/` | Help write SPARQL queries over SIB databases (UniProt, OMA, Rhea, Bgee...) |

Both are public, free, no authentication required.

---

## Add STRING-db to Mistral.ai Chat

1. Go to [chat.mistral.ai](https://chat.mistral.ai)
2. Click **Context** in the left sidebar > **Connectors** > **Add Connector** > **Custom MCP Connector**
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
2. Go to **Apps** in the left sidebar > ⚙️ top right > **Create app**
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

> ⚠️ All LLM client have a slightly different format to configure MCPs, but they are all really similar.

For [OpenCode](https://opencode.ai/download), open the user config file `~/.config/opencode/opencode.jsonc`:

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
    "openrouter": {
      "name": "OpenRouter",
      "npm": "@ai-sdk/openai-compatible",
      "models": {
        "gemma-4": {"name": "google/gemma-4-26b-a4b-it"}
      },
      "options": {
        "baseURL": "https://openrouter.ai/api/v1"
      }
    }
  }
}
```

---

## Connect a MCP server to proprietary agents

Claude Code stores it in `~/.claude.json`:

```sh
claude mcp add stringdb --transport http https://mcp.string-db.org/
```

Codex stores it in `~/.codex/config.toml`: 

```sh
codex mcp add stringdb -- --transport http https://mcp.string-db.org/
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

```sh
claude mcp add huggingface --transport http --header "Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/mcp
```

---

## MCP with stdio

Local MCP servers run as a subprocess via `stdio` (requires [`uv`](https://docs.astral.sh/uv) installed)

e.g. [PubMed MCP](https://github.com/grll/pubmedmcp) added to OpenCode:

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

A folder with a `SKILL.md` file, and additional docs or scripts

Requires to download the skill, e.g. to your laptop

Find skills for UniProt: [skills.sh](https://www.skills.sh)

Install using the npm `skills` package:

```sh
npx skills add https://github.com/google-deepmind/science-skills --skill uniprot-database
```

Or copy the skill folder in your agent local skill folder, e.g. `.claude/skills/uniprot-database/` (depends on the coding agent used)

---

## References

| Resource | URL |
| --- | --- |
| UniProt | <https://uniprot.org> |
| Rhea DB | <https://www.rhea-db.org> |
| OMA orthology | <https://omabrowser.org> |
| Bgee gene expression | <https://www.bgee.org> |
| STRING MCP server | <https://mcp.string-db.org> |
| Expasy MCP server | <https://expasy.org/chat> |
| SIB Swiss Institute of Bioinformatics | https://www.sib.swiss |

---

## Thank you
