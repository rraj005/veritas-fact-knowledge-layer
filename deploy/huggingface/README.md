---
title: Veritas Fact Knowledge Layer
emoji: 🔎
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Veritas — Fact Knowledge Layer

A domain-agnostic RAG system that extracts grounded facts from PDFs and detects
corroboration, contradiction, and context-reconcilable relationships across
documents. This Space runs the full app (REST API + web UI) from the repo's
`Dockerfile`.

**Using it:** open the Space, go to the **LLM Setup** panel, connect a provider
(OpenAI with your key, or Ollama), pick a model, then upload PDFs. The LLM is
bring-your-own-key — nothing is stored server-side.

> This file is the Hugging Face **Space** README. Its YAML front-matter tells
> HF to build the Docker image and serve the app on port 7860. When deploying,
> use this as the Space's `README.md`.
