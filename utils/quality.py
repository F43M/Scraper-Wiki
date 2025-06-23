"""Quality classification utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple


def classify_github_repo(repo: Dict) -> Tuple[str, str]:
    """Classify the quality of a GitHub repository.

    Parameters
    ----------
    repo: Dict
        Repository metadata containing at least ``stars`` and ``open_issues``.

    Returns
    -------
    Tuple[str, str]
        Quality level (``high``, ``medium`` or ``low``) and reason.
    """
    stars = repo.get("stars") or repo.get("stargazers_count", 0)
    issues = repo.get("open_issues") or repo.get("open_issues_count", 0)
    score = repo.get("quality_score")
    if score is None:
        score = stars / (issues + 1)
    has_tests = repo.get("has_tests") or repo.get("tests")

    if score >= 5:
        quality = "high"
        reason = "high star to issue ratio"
    elif score >= 2:
        quality = "medium"
        reason = "moderate star to issue ratio"
    else:
        quality = "low"
        reason = "low star to issue ratio"

    if has_tests:
        reason += " with tests"
    return quality, reason


def classify_stackoverflow_answer(answer: Dict) -> Tuple[str, str]:
    """Classify the quality of a StackOverflow answer.

    Parameters
    ----------
    answer: Dict
        Data containing ``score`` and optionally ``is_accepted``.

    Returns
    -------
    Tuple[str, str]
        Quality level and reason.
    """
    score = answer.get("score", 0)
    accepted = answer.get("is_accepted") or answer.get("accepted", False)

    if score >= 5 and accepted:
        return "high", "accepted answer with high score"
    if score >= 2:
        return "medium", "community upvoted"
    return "low", "low score"


def balance_quality(records: List[Dict]) -> List[Dict]:
    """Balance dataset records by quality level.

    The function keeps the same number of records for each quality
    class by truncating larger groups.
    """
    groups: Dict[str, List[Dict]] = {"high": [], "medium": [], "low": []}
    for rec in records:
        groups.setdefault(rec.get("quality", "medium"), []).append(rec)

    counts = [len(v) for v in groups.values() if v]
    if not counts:
        return records
    min_count = min(counts)
    balanced: List[Dict] = []
    for q in ["high", "medium", "low"]:
        balanced.extend(groups.get(q, [])[:min_count])
    return balanced


def generate_challenge_prompt(problems: List[str]) -> str:
    """Generate a Portuguese prompt challenging the user to fix the code.

    Parameters
    ----------
    problems: List[str]
        List of problems detected in the code.

    Returns
    -------
    str
        A short text in Portuguese describing the issues.
    """

    if not problems:
        return ""

    if len(problems) == 1:
        return f"Este código tem um bug: {problems[0]}. Corrija-o."

    joined = "; ".join(problems)
    return f"Este código tem alguns bugs: {joined}. Corrija-os."
