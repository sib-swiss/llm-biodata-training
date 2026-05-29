# 🧑‍🏫 Using Large Language Models for Biodata Exploration

[![Deploy to GitHub pages](https://github.com/sib-swiss/llm-biodata-training/actions/workflows/deploy.yml/badge.svg)](https://github.com/sib-swiss/llm-biodata-training/actions/workflows/deploy.yml)

Course description: https://www.sib.swiss/training/course/20260527_BAIBE

Pratical slides: [sib-swiss.github.io/llm-biodata-training](https://sib-swiss.github.io/llm-biodata-training)

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

To enable auth, initialize the SQLite db:

```sh
uv run init_db.py
```

## 🛠 Slides development

> Prerequisites: [NodeJS](https://nodejs.org/en/download)

Go to the `slides` folder:

```sh
cd slides
```

Install dependencies:

```sh
npm i
```

Deploy in development:

```sh
npm run dev
```

Build for production in the `dist` folder:

```sh
npm run build
```

Check production build:

```sh
npm run preview
```

Upgrade dependencies in `package.json`:

```sh
npm run upgrade
```

## 🎯 Deployment

This slide deck is set up to automatically deploy to GitHub Pages via GitHub Actions.
