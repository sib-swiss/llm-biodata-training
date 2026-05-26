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

## Setup GitHub Codespace

**GitHub Codespace** (already has Python, just install packages):

[github.com/codespaces](https://github.com/codespaces/) > **Blank** > **Use this template**

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) in github codespace:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Now you can use `uv` to manage projects and dependencies

---

## Setup dependencies

Create a `pyproject.toml`:

```toml
[project]
name = "biodata-mcp"
version = "0.0.1"
requires-python = "==3.13.*"
dependencies = [
    "mcp >=1.15.0",
    "requests >=2.34.2",
]
```

---

## Implement the MCP server

Create a `mcp_server.py` file. To build an MCP server in Python just decorate Python functions with `@mcp.tool()`:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="Biodata MCP",
    instructions="Do bio-cool stuff",
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
uv run mcp_server.py
```

The MCP endpoint is now at `http://localhost:8000/`.

Point any MCP-compatible client at it to use your tools:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "biodata": {
      "type": "remote",
      "url": "http://localhost:8000/",
      "enabled": true
    }
  }
}
```

---

## Questions

- Search for reviewed human TP53 proteins in UniProt
- What is the function of human BRCA1 and which diseases is it linked to?
- What is the sequence of the enzyme that catalyzes ATP hydrolysis, and what reactions involve it?
- Find the human enzyme for glucose phosphorylation and look up the Rhea reactions it catalyzes

---

## What's next?

Publish or deploy remotely?

- If just a wrapper that can run locally: publish as pip/npm package running through stdio transport
- If needs to run remotely (e.g. connected to a closed remote database): deploy as remote HTTP server

---

## Thank you
