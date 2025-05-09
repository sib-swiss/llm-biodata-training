## Introduction

Lightweight template for creating and deploying **[RevealJS](https://revealjs.com/)** slides using markdown and [vite](https://vite.dev/).

Simple, fast, and perfect for code-heavy presentations.

[Source code repository](https://github.com/vemonet/revealjs-template-markdown-vite)

---

## Outline

1. Setup dependencies
2. Run a script
3. Show an image

---

## Setup dependencies

[Install `uv`](https://docs.astral.sh/uv/getting-started/installation/) to easily handle dependencies and run scripts.

Create a `pyproject.toml` file with this content:

```toml
[project]
name = "some-project"
version = "0.0.1"
requires-python = "==3.12.*"
dependencies = [
    "httpx <=0.28.1",
]
```

And checkout the slide below.

----

Alternatively you can use inline dependencies at the start of your `.py` file:

```python
# /// script
# dependencies = [
#   "httpx <=0.28.1",
# ]
# ///
```

---

## Run a script

Create a `main.py` file in the same folder

```python
print("Hello world")
```

Run it with:

```sh
uv run main.py
```

---

## Show an image

<div class="r-stretch" style="display: flex;">
<div style="flex: 1;">

Some text on the left, with an image on the right

</div>
<div style="flex: 1;">
    <img src="logo.png" alt="Some image">
</div>
</div>
