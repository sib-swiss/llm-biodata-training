# 🧑‍🏫 Using Large Language Models for Biodata Exploration

[![Deploy to GitHub pages](https://github.com/sib-swiss/llm-biodata-training/actions/workflows/deploy.yml/badge.svg)](https://github.com/sib-swiss/llm-biodata-training/actions/workflows/deploy.yml)

Course description: https://www.sib.swiss/training/course/20260527_BAIBE

Pratical slides: [sib-swiss.github.io/llm-biodata-training](https://sib-swiss.github.io/llm-biodata-training)

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
