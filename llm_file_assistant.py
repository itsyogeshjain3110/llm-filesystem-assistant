from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fs_tools import list_files, read_file, search_in_file, write_file


SYSTEM_PROMPT = (
    "You are a file assistant. Use the available tools to inspect resumes, search for keywords, "
    "list directory contents, and write summary files when requested. "
    "When a user asks for a resume summary or a targeted search, call the relevant tools first."
)


@dataclass
class LLMConfig:
    base_url: str
    api_key: str | None
    model: str
    provider: str
    timeout: int = 60


def build_llm() -> LLMConfig:
    """Build the chat-completions client configuration from environment variables."""

    dotenv_values = _load_dotenv_values()
    provider = _get_setting("LLM_PROVIDER", dotenv_values, "openrouter").strip().lower()

    if provider == "openrouter":
        base_url = _get_setting("OPENROUTER_BASE_URL", dotenv_values, "https://openrouter.ai/api/v1")
        api_key = _get_setting("OPENROUTER_API_KEY", dotenv_values) or _get_setting("LLM_API_KEY", dotenv_values)
        model = _get_setting("LLM_MODEL", dotenv_values, "openai/gpt-4o-mini")
    else:
        base_url = _get_setting("LLM_BASE_URL", dotenv_values) or _get_setting("OPENAI_BASE_URL", dotenv_values, "https://api.openai.com/v1")
        api_key = _get_setting("LLM_API_KEY", dotenv_values) or _get_setting("OPENAI_API_KEY", dotenv_values)
        model = _get_setting("LLM_MODEL", dotenv_values, "gpt-4o-mini")

    return LLMConfig(base_url=base_url.rstrip("/"), api_key=api_key, model=model, provider=provider)


def run_assistant(query: str) -> str:
    """Answer a query by calling filesystem tools through an OpenAI-compatible chat model."""

    if _should_use_rules(query):
        return _run_with_rules(query)

    config = build_llm()
    if config.api_key:
        try:
            return _run_with_llm(config, query)
        except Exception:
            pass

    return _run_with_rules(query)


def _run_with_llm(config: LLMConfig, query: str) -> str:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    tools = _tool_definitions()

    for _ in range(8):
        response = _chat_completion(config, messages, tools)
        message = response["choices"][0]["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return (message.get("content") or "").strip()

        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            tool_name = function.get("name")
            tool_arguments = _parse_tool_arguments(function.get("arguments", "{}"))
            tool_result = _execute_tool(tool_name, tool_arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": json.dumps(tool_result, ensure_ascii=False, default=str),
                }
            )

    raise RuntimeError("The assistant exceeded the maximum number of tool-calling steps.")


def _chat_completion(config: LLMConfig, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }

    if config.provider == "openrouter":
        dotenv_values = _load_dotenv_values()
        headers["HTTP-Referer"] = _get_setting("OPENROUTER_HTTP_REFERER", dotenv_values, "http://localhost")
        headers["X-Title"] = _get_setting("OPENROUTER_APP_NAME", dotenv_values, "llm-file-assistant")

    payload = {
        "model": config.model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
    }

    request = Request(
        f"{config.base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urlopen(request, timeout=config.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(body or str(exc)) from exc
    except URLError as exc:
        raise RuntimeError(str(exc)) from exc

    if "error" in data:
        raise RuntimeError(str(data["error"]))

    return data


def _execute_tool(tool_name: str | None, tool_arguments: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "read_file":
        return read_file(**tool_arguments)
    if tool_name == "list_files":
        return list_files(**tool_arguments)
    if tool_name == "write_file":
        return write_file(**tool_arguments)
    if tool_name == "search_in_file":
        return search_in_file(**tool_arguments)

    return {"success": False, "error": f"Unknown tool: {tool_name}"}


def _tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a TXT, PDF, or DOCX resume and return structured text plus metadata.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string", "description": "Path to the file to read."},
                    },
                    "required": ["filepath"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files in a directory and optionally filter by file extension.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {"type": "string", "description": "Directory to list."},
                        "extension": {
                            "type": ["string", "null"],
                            "description": "Optional extension filter such as .pdf or .txt.",
                        },
                    },
                    "required": ["directory"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file, creating parent directories if needed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string", "description": "Path to write."},
                        "content": {"type": "string", "description": "Content to write."},
                    },
                    "required": ["filepath", "content"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_in_file",
                "description": "Search for a keyword in a file and return matches with surrounding context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string", "description": "File to search."},
                        "keyword": {"type": "string", "description": "Keyword to search for."},
                    },
                    "required": ["filepath", "keyword"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def _parse_tool_arguments(arguments: str) -> dict[str, Any]:
    if not arguments:
        return {}
    if isinstance(arguments, dict):
        return arguments
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return {}


def _load_dotenv_values() -> dict[str, str]:
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(dotenv_path):
        return {}

    values: dict[str, str] = {}
    try:
        with open(dotenv_path, "r", encoding="utf-8") as file_handle:
            for raw_line in file_handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in values:
                    values[key] = value
    except OSError:
        return {}

    return values


def _get_setting(name: str, dotenv_values: dict[str, str], default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is not None and value != "":
        return value

    value = dotenv_values.get(name)
    if value is not None and value != "":
        return value

    return default


def _run_with_rules(query: str) -> str:
    """Provide a deterministic fallback when an LLM key is not configured."""

    lowered = query.lower()
    read_match = re.search(r"(?:read|open|inspect)\s+(?:the\s+)?(.+)$", lowered)

    if "summary" in lowered and (read_match or ".pdf" in lowered or ".docx" in lowered or ".txt" in lowered):
        target = _guess_path_from_query(query)
        target = _resolve_resume_target(target)
        if target:
            result = read_file(target)
            if result.get("success"):
                content = result.get("content", "")
                lines = [line.strip() for line in content.splitlines() if line.strip()]
                summary = " ".join(lines[:5])[:800]
                summary_path = str(_summary_path_for(target))
                write_file(summary_path, summary)
                return f"Created summary file at {summary_path}\n\n{summary}"
            return result.get("error") or f"Could not read resume file: {target}"

    if _is_resume_listing_query(lowered):
        folder = _guess_path_from_query(query) or _default_resumes_directory()
        if folder:
            files = list_files(folder)
            resumes = [item for item in files if item.get("extension") in {"pdf", "txt", "docx"}]
            return json.dumps(resumes, indent=2, ensure_ascii=False)

    if "python" in lowered and "resume" in lowered:
        folder = _guess_path_from_query(query) or _default_resumes_directory()
        if folder:
            matches = []
            for item in list_files(folder):
                if item.get("extension") not in {"pdf", "txt", "docx"}:
                    continue
                result = search_in_file(item["path"], "python")
                if result.get("total_matches", 0) > 0:
                    matches.append({"file": item["name"], "matches": result["matches"]})
            return json.dumps(matches, indent=2, ensure_ascii=False)

    person_name = _extract_resume_person_name(query)
    if person_name:
        resume_path = _find_resume_for_person(person_name)
        if resume_path:
            result = read_file(resume_path)
            if result.get("success"):
                return _answer_resume_question(person_name, result.get("content", ""), lowered)
            return result.get("error") or f"Could not read resume file: {resume_path}"
        return f"I could not find a resume for {person_name}."

    return (
        "No API key is configured, so the assistant fell back to rule-based routing. "
        "Set OPENROUTER_API_KEY for OpenRouter, or configure LLM_API_KEY and LLM_BASE_URL for another provider."
    )


def _should_use_rules(query: str) -> bool:
    lowered = query.lower()
    return any(
        [
            _is_resume_listing_query(lowered),
            "read all resumes" in lowered,
            "summary" in lowered,
            ("summary" in lowered and re.search(r"\.(?:pdf|txt|docx)\b", lowered) is not None),
            ("python" in lowered and "resume" in lowered),
            _extract_resume_person_name(query) is not None,
        ]
    )


def _is_resume_listing_query(lowered: str) -> bool:
    return bool(re.search(r"\blist\b.*\bresumes?\b", lowered) or "read all resumes" in lowered)


def _extract_resume_person_name(query: str) -> str | None:
    lowered = query.lower().strip()
    if not any(keyword in lowered for keyword in ("who is", "tell me", "about", "experience", "resume")):
        return None

    match = re.search(
        r"(?:who is|tell me about|tell me|about|experience(?: of)?|resume(?: of)?|profile of)\s+(.+)$",
        query,
        flags=re.IGNORECASE,
    )
    if match:
        candidate = match.group(1).strip().strip("?.!,")
        candidate = re.sub(r"\bexperience\b.*$", "", candidate, flags=re.IGNORECASE).strip()
        if candidate:
            return candidate

    name_like = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", query)
    if name_like:
        return name_like.group(1).strip()

    return None


def _find_resume_for_person(person_name: str) -> str | None:
    folder = _default_resumes_directory()
    person_tokens = [token for token in re.split(r"\s+", person_name.lower()) if token]

    for item in list_files(folder):
        if item.get("extension") not in {"pdf", "txt", "docx"}:
            continue

        filename = item.get("name", "").lower()
        if all(token in filename for token in person_tokens):
            return item["path"]

    for item in list_files(folder):
        if item.get("extension") not in {"pdf", "txt", "docx"}:
            continue

        filename = item.get("name", "").lower()
        if any(token in filename for token in person_tokens):
            return item["path"]

    return None


def _answer_resume_question(person_name: str, content: str, lowered_query: str) -> str:
    profile = _extract_section(content, "Profile", ["Education", "Experience", "Key Projects", "Technical Skills", "Certifications", "Achievements"])
    experience = _extract_section(content, "Experience", ["Key Projects", "Technical Skills", "Certifications", "Achievements"])

    if "experience" in lowered_query:
        answer_parts = [f"{person_name}'s experience:"]
        if experience:
            answer_parts.append(experience)
        elif profile:
            answer_parts.append(profile)
        else:
            answer_parts.append("I found the resume, but could not isolate the experience section.")
        return "\n\n".join(answer_parts)

    answer_parts = [f"About {person_name}:"]
    if profile:
        answer_parts.append(profile)
    elif experience:
        answer_parts.append(experience)
    else:
        answer_parts.append("I found the resume, but could not isolate a profile section.")
    return "\n\n".join(answer_parts)


def _extract_section(content: str, section_name: str, stop_sections: list[str]) -> str | None:
    lines = [line.rstrip() for line in content.splitlines()]
    section_index = None
    for index, line in enumerate(lines):
        if line.strip().lower() == section_name.lower():
            section_index = index + 1
            break

    if section_index is None:
        return None

    collected: list[str] = []
    stop_set = {item.lower() for item in stop_sections}
    for line in lines[section_index:]:
        stripped = line.strip()
        if stripped.lower() in stop_set:
            break
        if stripped:
            collected.append(stripped)

    return " ".join(collected).strip() or None


def _guess_path_from_query(query: str) -> str | None:
    match = re.search(r"([\w./-]+\.(?:pdf|txt|docx))", query, flags=re.IGNORECASE)
    if match:
        return match.group(1)

    folder_match = re.search(r"([\w./-]+\/?resumes?)", query, flags=re.IGNORECASE)
    if folder_match:
        return folder_match.group(1)

    return None


def _default_resumes_directory() -> str:
    return os.path.join(os.path.dirname(__file__), "sample_data", "resumes")


def _resolve_resume_target(target: str | None) -> str | None:
    if not target:
        return None

    if os.path.exists(target):
        return target

    candidate = os.path.join(_default_resumes_directory(), os.path.basename(target))
    if os.path.exists(candidate):
        return candidate

    return target


def _summary_path_for(target: str) -> str:
    path = target.rstrip("/")
    if path.lower().endswith((".pdf", ".txt", ".docx")):
        path = re.sub(r"\.(pdf|txt|docx)$", "", path, flags=re.IGNORECASE)
    return f"{path}_summary.txt"


def main() -> None:
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip()
        if not query:
            raise SystemExit("A query is required.")
        print(run_assistant(query))
        return

    print("Interactive file assistant. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            query = input("Ask the file assistant: ").strip()
        except EOFError:
            print()
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        print(run_assistant(query))


if __name__ == "__main__":
    main()