from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any
import mimetypes


def read_file(filepath: str) -> dict:
    """Read a TXT, PDF, or DOCX resume and return structured content."""

    path = _expand_path(filepath)
    if not os.path.exists(path):
        return _error_response(path, "File not found.")
    if not os.path.isfile(path):
        return _error_response(path, "The provided path is not a file.")

    metadata = _file_metadata(path)
    extension = os.path.splitext(path)[1].lower()

    try:
        if extension == ".txt":
            content = _read_txt(path)
        elif extension == ".pdf":
            content = _read_pdf(path)
        elif extension == ".docx":
            content = _read_docx(path)
        else:
            return _error_response(
                path,
                f"Unsupported file format: {extension or 'unknown'}. Supported types are PDF, TXT, and DOCX.",
                metadata=metadata,
            )
    except Exception as exc:
        return _error_response(path, f"Failed to read file: {exc}", metadata=metadata)

    metadata["content_length"] = len(content)
    metadata["word_count"] = len(content.split()) if content else 0

    return {
        "success": True,
        "file_path": os.path.abspath(path),
        "file_name": os.path.basename(path),
        "file_type": extension.lstrip(".") or None,
        "content": content,
        "metadata": metadata,
        "error": None,
    }


def list_files(directory: str, extension: str = None) -> list:
    """List files in a directory and optionally filter by extension."""

    path = _expand_path(directory)
    if not os.path.exists(path) or not os.path.isdir(path):
        return []

    normalized_extension = _normalize_extension(extension)
    files = []

    for child_name in sorted(os.listdir(path), key=str.lower):
        child_path = os.path.join(path, child_name)
        if not os.path.isfile(child_path):
            continue

        child_extension = os.path.splitext(child_name)[1].lower()
        if normalized_extension and child_extension != normalized_extension:
            continue

        stat_result = os.stat(child_path)
        files.append(
            {
                "name": child_name,
                "path": os.path.abspath(child_path),
                "size": stat_result.st_size,
                "modified_date": datetime.fromtimestamp(stat_result.st_mtime, tz=timezone.utc).isoformat(),
                "extension": child_extension.lstrip(".") or None,
            }
        )

    return files


def write_file(filepath: str, content: str) -> dict:
    """Write content to a file and create parent directories when needed."""

    path = _expand_path(filepath)

    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file_handle:
            file_handle.write(content)
        stat_result = os.stat(path)
        return {
            "success": True,
            "file_path": os.path.abspath(path),
            "file_name": os.path.basename(path),
            "bytes_written": len(content.encode("utf-8")),
            "metadata": {
                "size": stat_result.st_size,
                "modified_date": datetime.fromtimestamp(stat_result.st_mtime, tz=timezone.utc).isoformat(),
            },
            "error": None,
        }
    except Exception as exc:
        return _error_response(path, f"Failed to write file: {exc}")


def search_in_file(filepath: str, keyword: str) -> dict:
    """Search for a keyword in a file and return matches with surrounding context."""

    read_result = read_file(filepath)
    if not read_result.get("success"):
        return {
            "success": False,
            "file_path": read_result.get("file_path"),
            "keyword": keyword,
            "matches": [],
            "total_matches": 0,
            "error": read_result.get("error") or "Failed to read file.",
        }

    content = read_result.get("content", "")
    lines = content.splitlines()
    matches = []
    keyword_lower = keyword.lower()

    for index, line in enumerate(lines):
        if keyword_lower not in line.lower():
            continue

        start_index = max(0, index - 1)
        end_index = min(len(lines), index + 2)
        matches.append(
            {
                "line_number": index + 1,
                "line": line,
                "context_before": "\n".join(lines[start_index:index]),
                "context_after": "\n".join(lines[index + 1:end_index]),
                "context": "\n".join(lines[start_index:end_index]),
            }
        )

    return {
        "success": True,
        "file_path": read_result.get("file_path"),
        "keyword": keyword,
        "matches": matches,
        "total_matches": len(matches),
        "error": None,
    }


def _normalize_extension(extension: str | None) -> str | None:
    if not extension:
        return None
    normalized = extension.strip().lower()
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    return normalized


def _file_metadata(path: str) -> dict[str, Any]:
    stat_result = os.stat(path)
    return {
        "absolute_path": os.path.abspath(path),
        "file_name": os.path.basename(path),
        "extension": os.path.splitext(path)[1].lower().lstrip(".") or None,
        "size_bytes": stat_result.st_size,
        "modified_date": datetime.fromtimestamp(stat_result.st_mtime, tz=timezone.utc).isoformat(),
        "mime_type": mimetypes.guess_type(path)[0],
    }


def _error_response(path: str, message: str, metadata: dict | None = None) -> dict:
    return {
        "success": False,
        "file_path": os.path.abspath(path) if os.path.exists(path) else path,
        "file_name": os.path.basename(path),
        "file_type": os.path.splitext(path)[1].lower().lstrip(".") or None,
        "content": "",
        "metadata": metadata or {},
        "error": message,
    }


def _read_txt(path: str) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as file_handle:
                return file_handle.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as file_handle:
        return file_handle.read()


def _read_docx(path: str) -> str:
    try:
        import docx
    except Exception as exc:
        raise RuntimeError("python-docx is required to read DOCX files") from exc

    document = docx.Document(path)
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    return "\n".join(paragraphs)


def _read_pdf(path: str) -> str:
    try:
        import PyPDF2
    except Exception as exc:
        raise RuntimeError("PyPDF2 is required to read PDF files") from exc

    chunks = []
    with open(path, "rb") as file_handle:
        reader = PyPDF2.PdfReader(file_handle)
        for page in reader.pages:
            extracted = page.extract_text() or ""
            extracted = extracted.strip()
            if extracted:
                chunks.append(extracted)
    return "\n\n".join(chunks)


def _expand_path(path: str) -> str:
    return os.path.expanduser(path)

