---
title: Chain of Thoughts Reasoning Demo
emoji: 🤗
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.9.0"
python_version: "3.13"
app_file: app.py
pinned: false
---

# Chain of Thoughts Reasoning Demo

Chain of Thoughts Reasoning Demo is a lightweight Gradio demo for running a reasoning-focused causal language model in a Hugging Face Space.
Qwen3-0.6-Base is supervised fine-tuned on a modified openai/gsm8k dataset, and the demo allows users to interact with the model by inputting prompts and receiving both the final answer and the extracted chain of thought.

## Overview

- Single-text input Gradio app
- Displays the final answer and the extracted chain of thought separately
- Loads the model from Hugging Face Hub: `Blankyy/reasoning-math-model`
- [Hugging Face space link](https://huggingface.co/spaces/Blankyy/fine-tuned-cot-qwen-base)

## Repository structure

- `app.py` — Gradio app entry point
- `pyproject.toml` — Python project metadata and dependencies
- `README.md` — project overview and deployment notes
- `cot_reasoning.ipynb` - Notebook used for training the model.
## Local setup

1. Create and activate a Python environment.
2. Install the project dependencies.
3. Run the app with Python.

## Hugging Face Spaces

This repository is ready to be used as a Gradio Space.

### Space settings

- SDK: Gradio
- Python version: 3.13 or newer
- Entry point: `app.py`

### Required dependencies

The app uses:

- gradio
- torch
- transformers

### Notes

- The model is loaded directly from the Hugging Face Hub at startup.
- If the Space is running on CPU, inference may be slower.
- For GPU-enabled Spaces, the app will automatically use CUDA when available.

## Usage

Type a prompt into the input box and submit it. The app returns:

- the final answer
- the extracted chain of thought, when available

## License

Add a license here if you plan to publish the repository publicly.
