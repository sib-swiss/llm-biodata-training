## Creating your own MCP server

Build MCP tools to expose biodata APIs to any LLM client: coding agents, or your own webapp.

---

## Outline

1. Design the tools and databases to expose
2. Implement the MCP server
3. Key design patterns
4. Run and test

---

## What do you need to expose?

Is there a resource you would like to expose to agents?

Can your MCP server just be a wrapper around an existing API?

Or does it need to deploy a whole database with indexing and maintenance?

---

## Databases we will expose

| Database | URL | Description |
| --- | --- | --- |
| **UniProt** | [uniprot.org](https://www.uniprot.org) | World's leading protein knowledgebase with 570k+ reviewed Swiss-Prot entries and 250M+ TrEMBL sequences. Each entry contains protein function, taxonomy, sequence, subcellular location, post-translational modifications, disease associations, and cross-references to 100+ databases. |
| **Rhea** | [rhea-db.org](https://www.rhea-db.org) | Expert-curated database of 15k+ biochemical reactions covering metabolism, transport, and biosynthesis. Reactions are described using ChEBI chemical identifiers and linked to UniProt enzymes, allowing precise querying of which proteins catalyze which reactions. |

---

## Implement the MCP server

Create a `mcp_server.py` file. To build an MCP server in Python just decorate Python functions with `@mcp.tool()`:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="Biodata MCP",
    instructions="Do cool stuff",
    streamable_http_path="/",
)

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers together."""
    return a + b

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

---

## Key design patterns for MCP tools

1. **Rich docstrings are the LLM's only interface**

The LLM reads the docstring to decide *when* and *how* to call the tool. A weak docstring means wrong tool selection.

1. **Return structured dicts, not plain strings**

```python
# Good: LLM can reason over fields
return {"accession": "P04637", "gene": "TP53", "diseases": [...], "go_terms": [...]}

# Bad: harder to parse, wastes tokens
return "P04637 is TP53, involved in Li-Fraumeni syndrome..."
```

1. **Truncate large payloads**

```python
"variants": variants[:50]
"go_terms": go_terms[:10]
"sequence": sequence[:2000]
```

---

## Tools to implement

**uniprot_search**(query, organism, reviewed_only, max_results)

```txt
GET https://rest.uniprot.org/uniprotkb/search
    ?query=TP53 AND organism_name:Homo sapiens AND reviewed:true
    &format=json&fields=accession,gene_names,protein_name,organism_name,length,reviewed
```

**uniprot_get_entry**(accession)

```txt
GET https://rest.uniprot.org/uniprotkb/{accession}?format=json
```

**uniprot_get_sequence**(accession)

```txt
GET https://rest.uniprot.org/uniprotkb/{accession}.fasta
```

**rhea_search_reactions**(query, max_results)

```txt
GET https://www.rhea-db.org/rhea
    ?query=ATP hydrolysis&format=json&columns=rhea-id,equation,chebi-id&limit=8
```

---

## Run your MCP server

Start the server:

```sh
# Codespace
python mcp_server.py

# Local with uv
uv run --env-file .env mcp_server.py
```

The MCP endpoint is now at `http://localhost:8000/mcp`.

Point any MCP-compatible client at it to use your tools:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "biodata": {
      "type": "remote",
      "url": "http://localhost:8000/mcp",
      "enabled": true
    }
  }
}
```

---

## Test questions

- Search for reviewed human TP53 proteins in UniProt
- What is the function of human BRCA1 and which diseases is it linked to?
- Find the human enzyme for glucose phosphorylation
- What biochemical reactions involve ATP hydrolysis in Rhea?

---

## What's next?

Publish or deploy:

- If just a wrapper that can run locally: publish as pip/npm package running through stdio transport
- If needs to run remotely (e.g. connected to a closed remote database): deploy as remote HTTP server

Then this MCP server in:

- Any MCP-compatible chat UI (Mistral Le Chat, ChatGPT)
- Coding agents (Claude Code, OpenCode, Cursor)
- A Python Chainlit webapp (next section)

---

## Thank you
