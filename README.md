# Grad COT

Grad COT is a lightweight Gradio demo for running a reasoning-focused causal language model in a Hugging Face Space.

## Overview

- Single-text input Gradio app
- Displays the final answer and the extracted chain of thought separately
- Loads the model from Hugging Face Hub: `Blankyy/reasoning-math-model`

## Repository structure

- `app.py` — Gradio app entry point
- `pyproject.toml` — Python project metadata and dependencies
- `README.md` — project overview and deployment notes

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
