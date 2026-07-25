# LLM Filesystem Assistant

Small Python assistant for reading resume files, listing directory contents, searching within files, and writing summary files. It uses OpenRouter by default for chat-completions tool calling and falls back to rule-based behavior when no API key is configured.

## What it includes

- `fs_tools.py` for file read, list, write, and search helpers
- `llm_file_assistant.py` for OpenRouter-based orchestration
- `sample_data/resumes/` with dummy resume files you can use for testing

## Requirements

- Python 3.12+ recommended
- An OpenRouter API key in `OPENROUTER_API_KEY` for the default setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Put your configuration in `.env` or export the variables in your shell:

```bash
export OPENROUTER_API_KEY="your-key"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
export LLM_MODEL="openai/gpt-4o-mini"
```

If you prefer a different provider, set `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` in the same `.env` file.

Optional environment variables:

- `OPENROUTER_HTTP_REFERER` for the request header
- `OPENROUTER_APP_NAME` for the request header

To use another OpenAI-compatible provider, set these instead of the OpenRouter values:

```bash
export LLM_PROVIDER="openai"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_API_KEY="your-key"
export LLM_MODEL="gpt-4o-mini"
```

## Usage

Run the assistant in interactive mode:

```bash
python llm_file_assistant.py
```

Then type multiple questions in the same session. Use `exit` or `quit` to stop.

You can also run a one-off query from the command line:

```bash
python llm_file_assistant.py "list resumes in sample_data/resumes"
python llm_file_assistant.py "summarize sample_data/resumes/jane_doe_resume.txt"
python llm_file_assistant.py "search sample_data/resumes for python"
```

You can also import the helper functions directly:

```python
from fs_tools import read_file, list_files, write_file, search_in_file
```

## Sample data

Dummy resumes are provided in `sample_data/resumes/` as `.pdf` files:

- `jane_doe_resume`
- `alex_kim_resume`
- `sam_patel_resume`
- `YogeshJainResume`
- `ArjunSharmaResume`


These files are easy to inspect, search, and summarize during development.

## Notes

- `fs_tools.py` supports `.txt`, `.pdf`, and `.docx` files.
- PDF and DOCX support requires the packages listed in `requirements.txt`.
- OpenRouter is the default provider, but any OpenAI-compatible endpoint can be used with `LLM_PROVIDER`, `LLM_BASE_URL`, and `LLM_API_KEY`.
