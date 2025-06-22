import re
import textwrap


def normalize_indentation(code: str) -> str:
    """Return ``code`` with common indentation removed."""
    dedented = textwrap.dedent(code)
    lines = [line.rstrip() for line in dedented.splitlines()]
    return "\n".join(lines).strip()


def remove_comments(code: str, language: str) -> str:
    """Return ``code`` without comments for ``language``."""
    patterns = []
    lang = language.lower()
    if lang in {"python", "py"}:
        patterns = [r"#.*$", r'""".*?"""', r"'''(.|\n)*?'''"]
    elif lang in {"javascript", "js", "java", "c", "cpp", "go", "php", "rust"}:
        patterns = [r"//.*", r"/\*.*?\*/"]
    else:
        patterns = [r"#.*", r"//.*", r"/\*.*?\*/"]
    text = code
    for pat in patterns:
        flags = re.DOTALL
        if pat == r"#.*$":
            flags = re.MULTILINE
        text = re.sub(pat, "", text, flags=flags)
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def detect_programming_language(code: str) -> str:
    """Guess the programming language of ``code`` using simple heuristics."""
    heuristics = {
        "python": ["def ", "import ", "print("],
        "javascript": ["function ", "console.log", "var ", "let ", "const "],
        "java": ["public class", "System.out.println"],
        "c": ["#include", "printf("],
        "cpp": ["std::", "#include <iostream"],
        "php": ["<?php", "echo "],
        "ruby": ["end", "puts "],
        "go": ["package main", "fmt."],
        "rust": ["fn ", "::"],
    }
    lowered = code.lower()
    for lang, keys in heuristics.items():
        if any(k.lower() in lowered for k in keys):
            return lang
    return "unknown"
