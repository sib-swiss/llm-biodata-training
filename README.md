# 🧑‍🏫 RevealJS Slides Markdown Template

[![Deploy to GitHub pages](https://github.com/vemonet/revealjs-template-markdown-vite/actions/workflows/deploy.yml/badge.svg)](https://github.com/vemonet/revealjs-template-markdown-vite/actions/workflows/deploy.yml)

A lightweight and efficient template for creating and deploying **[RevealJS](https://revealjs.com/)** presentations using **markdown** and **[vite](https://vite.dev/)**. This setup is particularly useful for presentations that include code snippets.

> Checkout the [demo](https://vemonet.github.io/revealjs-template-markdown-vite).

## 🪄 Features

- 📝 **Write slides in Markdown** – Simple and fast editing.

- 🌐 **Auto-deploy as a static site** – Works seamlessly with GitHub Pages or any static hosting.

- 💡 **Syntax highlighting out of the box** – Supports most programming languages.

- 📋 **Copy button for code blocks** – Make it easy for your audience to grab code.

## 🎨 Getting Started

### Edit Your Slides

Modify **`public/slides.md`** to create your presentation:

- Use `---` to separate horizontal slides.
- Use `----` to create vertical slides (subslides).

### Customize the Look

Edit **`index.html`** to:

- Modify styling in the `<style>` block.
- Add RevealJS plugins for extra functionality.

> [!TIP]
>
> We recommend [Typora](https://typora.io/) for a smooth Markdown editing experience.

## 🛠 Development

> Prerequisites: [NodeJS](https://nodejs.org/en/download)

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

This template is set up to automatically deploy to GitHub Pages via GitHub Actions. You can also deploy manually to any static hosting provider.
