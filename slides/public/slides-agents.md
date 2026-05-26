## Coding Agents for biodata exploration

Introduction to using coding agents to interact with biological data and databases

> Vincent Emonet · SIB Swiss Institute of Bioinformatics

---

## Outline

1. Introduction: what are LLM agents?
2. Usage examples
3. LLM harnesses
4. Standards
5. Current limitations
6. Recommendations

---

## Biodata requires code

Biological data and scientific research has increasingly required code to access and process data:

- **Databases** (UniProt, Ensembl) expose REST APIs, endpoints, querying them means writing Python, R, SQL, or SPARQL
- **Formats** (FASTQ, VCF, PDB) require parsers and domain-specific libraries (Biopython, Bioconductor)
- **Analysis pipelines** (variant calling, protein structure prediction, pathway enrichment) needs to be coded

> Some even says that ["All Biology is Computational Biology"](https://doi.org/10.1371/journal.pbio.2002050)

But biologists are not computer scientists by training.

---

## Where LLMs close the gap

LLMs can assist in two complementary ways:

| Task | What the LLM does |
| --- | --- |
| **Code generation** | Write code and queries, parsing scripts, and analysis pipelines to access, filter, and transform data |
| **Interpretation** | Summarize retrieved records, explain biological significance |

It can write code, parse the result, and explain what it means, closing the loop between data retrieval and biological insight.

Research requires exact provenance and traceability, which is not guaranteed by regular LLM inference, tools to connect to trustworthy databases are needed.

---

## What are LLM agents?

> An LLM agent runs tools in a loop to achieve a goal

In practice:

- Generic tools like run a given bash command, and additional ones
- Coding agents often also have access to diagnostics from a language client or IDE
- Context management: summarize when conversation gets too long
- Planning: think step by step, todo lists

---

## Timeline

| Era                 | Focus Period   | Primary Technique                                | Goal                                                     | Output Behavior                                         |
| ------------------- | -------------- | ------------------------------------------------ | -------------------------------------------------------- | ------------------------------------------------------- |
| **1. Pre-training** | 2018 – 2022    | Self-Supervised Learning (next-token prediction) | Learn language, facts, and patterns from books & web.    | Predicts the next logical word (advanced autocomplete). |
| **2. Alignment**    | 2022 – 2023    | SFT & RLHF (Human feedback loops)                | Shape the model into a helpful conversational assistant. | Responds directly to instructions and prompts.          |
| **3. Tool Use**     | 2023 – Present | Function Calling & Agentic Fine-tuning           | Connect the LLM to real-world software, data.            | Writes code to trigger tools when it hits a limitation. |

---

## AI is just a tool

> There is *zero* point in talking about AI slop. That's just plain stupid. [...]
>
> As I said in private elsewhere, I do *not* want any kernel  development documentation to be some AI statement. We have enough people on both sides of the "sky is falling" and "it's going to revolutionize  software engineering", I don't want some kernel development docs to take either stance.
>
> It's why I strongly want this to be that "just a tool" statement.

\- [Linus Torvalds](https://lore.kernel.org/lkml/CAHk-=wg0sdh_OF8zgFD-f6o9yFRK=tDOXhB1JAxfs11W9bX--Q@mail.gmail.com/)

[github.com/torvalds/AudioNoise](https://github.com/torvalds/AudioNoise?tab=readme-ov-file#another-silly-guitar-pedal-related-repo)

---

## Human-directed agentic engineering

[Ladybird browser adopts Rust, with help from AI](https://ladybird.org/posts/adopting-rust)

> I used Claude Code and Codex for the translation. **This was human-directed, not autonomous code generation**. I decided what to port, in what order, and what the Rust code should look like. It was **hundreds of small prompts**, steering the agents where things needed to go.
>
> After the initial translation, I ran multiple passes of adversarial review, asking different models to analyze the code for mistakes and bad patterns. [...]

---

## Blind AI trust is a risk

> I didn’t write a single line of code for @moltbook. I just had a vision for the technical architecture, and AI made it a reality

\- Moltbook creator

The [Supabase URL and API key were found](https://www.wiz.io/blog/exposed-moltbook-database-reveals-millions-of-api-keys) in clear in the JavaScript code sent to users browsers

Cursor + Claude Opus 4.6 [dropped the production database of a startup (PocketOS)](https://www.theguardian.com/technology/2026/apr/29/claude-ai-deletes-firm-database), building fleet management software for rental car companies

---

## The agent harness

A raw LLM only predicts text. A **harness** is everything wrapped around the model that turns it into an agent: the loop, the tools, the context, the permissions.

It decides:

- What **files and context** the LLM can see
- What **tools** it can call (and with what permissions)
- How **memory** persists across turns and sessions
- How **long-running tasks** are managed (planning, todos, sub-agents)

> Same model, different harness, completely different capabilities.

---

## The spectrum · constrained to full control

| Harness | Files | Tools | Control |
| --- | --- | --- | --- |
| **ChatGPT / Claude.ai webapps** | Uploaded files only | Web search, code interpreter sandbox, remote MCP | Vendor-controlled, no host access |
| **Self-hosted chat webapp** ([Open WebUI](https://openwebui.com), [LibreChat](https://librechat.ai)) | Uploaded files, local files | MCP, custom tools you wire in | You control the model and the tools |
| [Google **NotebookLM**](https://notebooklm.google.com) | Uploaded documents | Q&A, audio overview | Locked to a corpus |
| **[LLM wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)** (Karpathy) | Local folder of markdown notes | Search notes, agent writes/updates files | Self-hosted, user-controlled |
| **Coding agents** (Cursor, Claude Code, Codex, GitHub Copilot) | Local project files, full filesystem | Bash, MCP, skills, language server | User-controlled on the machine |
| **Remote agents** (OpenClaw, Hermes, Pi) | Local files, shared drives | Bash, MCP, skills, scheduled jobs | Runs autonomously |

More power means more capability **and** more risk.

---

## Coding agents UI approaches

#### CLI

- ✅ Easy to install and run on a remote machine
- ✅ Works for people coding in neovim or over SSH
- ⚠️ TUIs are not made for complex reactive UIs

#### Desktop app

- ✅ Can be optimized for tasks beyond code (writing, research, ops)
- ✅ Agent runs separately from your editor
- ⚠️ Risks reinventing a lightweight IDE

#### Integrated in IDE

- ✅ Same diagnostics as you (language server, linter)
- ✅ Easy to iterate and edit generated code
- ✅ Fits existing dev setup

---

## Coding agents landscape

| Agent                 | CLI  | Desktop<br />app | IDE  | Open<br />source | Free<br />tier | Tech        |
| --------------------- | ---- | ---------------- | ---- | ---------------- | -------------- | ----------- |
| Cursor                | ✅    | ✅                | ✅    |                  | ☑️              | VSCode fork |
| VSCode GitHub Copilot | ✅    |                  | ✅    | ☑️                | ☑️              | VSCode      |
| Claude Code           | ✅    | ✅                | ☑️    |                  |                | TS          |
| OpenAI Codex          | ✅    | ✅                | ☑️    | ☑️                |                | Rust        |
| Google Antigravity    | ✅    |                  | ✅    |                  | ☑️              | VSCode fork |
| OpenCode              | ✅    | ✅                | ☑️    | ✅                |                | TS          |
| Goose                 | ✅    | ✅                |      | ✅                |                | Rust/TS     |
| Mistral Vibe          | ✅    |                  |      | ✅                |                | Python      |

---

## Coding agents · quick picks

- **Best tokens for money**: Claude Code, or OpenAI Codex
- **Best IDE integration**: Cursor, GitHub Copilot, Antigravity
- **Best open source**: OpenCode

---

## What makes a harness work · open standards

A harness is only as useful as the context and tools you can plug into it.

A few standards are emerging:

- [**AGENTS.md**](https://agents.md/) / [CLAUDE.md](https://docs.anthropic.com/en/docs/claude-code/memory) · project-level instructions the agent loads on every run
- [**MCP** Model Context Protocol](https://modelcontextprotocol.io) · expose tools and resources to LLMs in a standard way
- [**Skills**](https://agentskills.io) · markdown instructions and scripts the agent can pull in on need
- [Spec-based development](https://github.com/github/spec-kit) · write specs in markdown, the agent implements them

Communication protocols:

- [A2A](https://a2a-protocol.org/latest/) · agent-to-agent protocol
- [AG-UI](https://ag-ui.com) · agent-to-UI exchange protocol

Linux Agentic AI foundation: [aaif.io](https://aaif.io)

---

## AGENTS.md · project instructions

[AGENTS.md](https://agents.md/), or CLAUDE.md, tells the agent what it needs to know on every run: coding conventions, where to find docs.

- User level, for all projects, e.g. `~/.claude/CLAUDE.md`
- Project level, at the root of your project folder

Poorly designed instructions can **reduce** performance: too long, too vague, or contradictory instructions confuse the model.

[arxiv:2602.11988](https://arxiv.org/pdf/2602.11988) | [arxiv:2601.20404](https://arxiv.org/pdf/2601.20404)

```markdown
- Read existing files before writing code
- Keep solutions simple and direct. No over-engineering
- If unsure: say so. Never guess or invent file paths and function names
- Always check for lint and type errors before ending a task, and fix if there are some errors related to your changes
- When relevant run scripts to evaluate if your changes worked. For python use `uv run`, for JS use `npm run`
```

---

## MCP · expose tools to the LLM

An MCP server exposes:

| Primitive | Description | Example |
| --- | --- | --- |
| **`tools`** | Take typed arguments, execute an action, return results | `{"id": "TP53", "species": "9606"}` -> interaction partners |
| `resources` | Data the agent can read without side effects | UniProt entry, local FASTA file |
| `prompts` | Reusable prompt templates the agent can invoke | "Summarize this protein in one sentence" |

Using a transport:

- `stdio` · running the server locally as a subprocess
- `streamable-http` · deploying it as a remote HTTP endpoint

The agent sees only the tool name, docstring, and typed parameters, never the implementation.

---

## MCP server · example

```python
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

mcp = FastMCP(
    name="SIB MCP", dependencies=["mcp", "pydantic"],
    instructions="Tools for biodata exploration.",
)

@mcp.tool()
async def search_datasets(search_input: str, update_date: str | None = None) -> list[SearchResult]:
    """Search for datasets relevant to the user question.

    Args:
        search_input: Natural language search input
        update_date: Optional last update date in yyyy-MM-dd

    Returns:
        Relevant datasets
    """
    return get_relevant_datasets(search_input, update_date)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

---

## MCP registries

Official registry effort: [registry.modelcontextprotocol.io](https://registry.modelcontextprotocol.io)

GitHub curated MCP registry: [github.com/mcp](https://github.com/mcp)

Biodata-focused registry: [biocontext.ai/registry](https://biocontext.ai/registry)

---

## Skills · on-demand instructions

A collection of markdown files with YAML frontmatter, optionally bundled with scripts.

Each skill has a short description. The agent reads only the description by default and pulls in the full skill body **only when it decides it is relevant**.

Skills can be pure documentation ("how we deploy") or contain executable scripts.

Browse at [skills.sh](https://skills.sh)

> Skills **run on your machine**. MCP servers can run remotely, with the execution happening on the server.

---

## Skills · example

Skill to roll dice

````markdown
/---
name: roll-dice
description: Roll dice with true randomness. Use when asked to roll a die (d6, d20, etc.), generate a random dice roll.
/---

To roll a die, use the following command that generates a random number from 1
to the given number of `$SIDES`:

```sh
shuf -i 1-$SIDES -n 1
```
````

---

## Current limitations · context window

Tools and skills clutter the context window if you have too many.

- Every tool of every connected MCP server is fully loaded into context. Performance degrades past ~100 tools.
- Skills are pulled in only when the LLM decides they are relevant, lighter on context, but the LLM sometimes fails to trigger a skill it actually needs.
- Long conversations get auto-summarized, useful, but lossy.

---

## Current limitations · security risks

Skills or MCP server can easily be manipulated to [steal your credentials](https://www.koi.ai/blog/clawhavoc-341-malicious-clawedbot-skills-found-by-the-bot-they-were-targeting) (like any pip/npm packages)

Be wary of what you are installing

📍 Pin a specific version or commit hash to prevent supply chain attacks

> Limited knowledge on agentic engineering itself

---

## Current limitations · knowledge cutoff

Treat the LLM as a human-computer interface, not a trustworthy knowledge base.

✅ Good with stable, long-lived standards

> "What does HTTP 404 mean?"

⚠️ Bad out of the box with fast-moving libraries

> "What's the function for Y in the latest release of X?"

Give it access to up-to-date docs through tools, or copy the relevant context into your project.

---

## Current limitations · cost and reliability

LLM compute is not free.

- Under more load when the US is awake (after 15:00 CET), degraded responses
- Cheap token pricing might change

Pick the right model for the task:

- Small/cheap model · "generate a schema for this dict"
- Large/expensive model · "implement this feature, improve performance"

---

## Different usage profiles

- Use an LLM occasionally for help
- Use the LLM to write most of the code, but still read and understand it
- Delegate everything to the agent, no longer read the code

Even occasional users should consider an agent harness, it gives the LLM far more context than a chat webapp would.

---

## Do · setup

- Clearly define the problem, provide tests to pass
- Use explicit types and linters (eslint, ruff, mypy)
- Watch out for custom zsh config (e.g. oh-my-zsh quirks)
- Explore tools for your use-case, e.g. [chrome-devtools-mcp](https://github.com/ChromeDevTools/chrome-devtools-mcp)
- **Commit** before any major change so you can `git diff` or revert cleanly
- Back up your machine (git, dropbox), keep offline DB backups

---

## Do · give it a target to hit

**Test-driven development** · write the test first, let the agent make it pass:

> "Here are 3 failing test cases for a FASTA parser. Make them pass without changing the tests."

The agent has an unambiguous success target. It verifies by running tests.

---

## Do · keep prompts small

**Small, scoped prompts** · one thing at a time:

> "Add a `GET /proteins` endpoint that returns this list from the existing SQLite schema [schema]. No auth yet."

Not: "Build me a complete REST API with auth, CRUD, rate limiting, and tests."

---

## Do · give it context

**Paste the actual error** when stuck · avoid "it doesn't work":

> "This function raises `KeyError: 'gene_names'` on this input [input]. Fix it."

**Copy relevant docs** into the project when the library changes fast:

> Copy the relevant section of the changelog or API docs into a file in the repo.

---

## Don't · over-delegation

Not reading the generated code. Not running the tests. Not doing security review.

The agent looks confident even when it is wrong. 

You are the last line of defense.

---

## Don't · let it touch irreversible actions unsupervised

Actions that are hard to undo - dropping a database table, pushing to production, deleting files - need a human in the loop.

- Revoke broad permissions after a session; grant only what the task needs
- Never run with `--dangerously-skip-permissions` (or equivalent) before you understand every tool the agent will call
- Keep offline backups of anything the agent can write to

---

## Don't · treat it as a knowledge oracle

The model sounds authoritative even when it is hallucinating.

- Cross-check any biological claim against a primary database (UniProt, Ensembl, PubMed)
- Prefer tools that return live data (MCP servers, web search) over recalled "knowledge"
- Ask it to cite sources; if it cannot, treat the claim as a starting point, not a conclusion

---

## Thank you

- [Agentic Engineering Patterns](https://simonwillison.net/2026/Feb/23/agentic-engineering-patterns/)
- [Augmented Coding Patterns](https://lexler.github.io/augmented-coding-patterns/)
- [BioContext MCP registry](https://biocontext.ai/registry) - MCP servers for life sciences
- [MCP official registry](https://registry.modelcontextprotocol.io)
- [skills.sh](https://skills.sh) - community skill library
