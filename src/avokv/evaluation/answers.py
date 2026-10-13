from __future__ import annotations

import re


def _balanced_braces(text: str, opening: int) -> str:
    depth = 0
    collected = []
    for character in text[opening:]:
        if character == "{":
            depth += 1
            if depth > 1:
                collected.append(character)
        elif character == "}":
            depth -= 1
            if depth == 0:
                return "".join(collected)
            collected.append(character)
        elif depth:
            collected.append(character)
    return ""


def extract_boxed(text: str) -> str | None:
    """Extract the final balanced ``\\boxed{...}`` expression."""
    starts = [match.start() for match in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        starts = [match.start() for match in re.finditer(r"boxed\s*\{", text)]
    if not starts:
        return None
    opening = text.find("{", starts[-1])
    answer = _balanced_braces(text, opening)
    return answer if answer else None


def normalize_answer(answer: object) -> str:
    """Apply the normalization shared by predictions and references."""
    text = str(answer).strip()
    boxed = extract_boxed(text)
    if boxed is not None:
        text = boxed

    text = text.replace("\n", "").rstrip(".")
    text = text.replace("\\!", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    text = text.replace("\\$", "").replace("$", "")
    text = text.replace("\\%", "").replace("%", "")
    text = re.sub(r"\\text\{(.*?)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{(.*?)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    if len(text.split("=")) == 2 and len(text.split("=")[0]) <= 2:
        text = text.split("=", 1)[1]
    return text.strip().rstrip("/")


def extract_answer(text: str) -> str:
    """Extract a final answer using the paper runner's output convention."""
    boxed = extract_boxed(text)
    if boxed is not None:
        return normalize_answer(boxed)

    lower = text.lower()
    for marker in ("final answer:", "final answer is", "the answer is"):
        position = lower.rfind(marker)
        if position >= 0:
            suffix = text[position + len(marker) :].strip()
            first_line = suffix.splitlines()[0] if suffix else ""
            return normalize_answer(first_line)

    numbers = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", text.replace(",", ""))
    return normalize_answer(numbers[-1]) if numbers else ""


def reference_answer(row: dict) -> str:
    if row.get("answer") not in (None, ""):
        return normalize_answer(row["answer"])
    if row.get("solution") not in (None, ""):
        extracted = extract_boxed(str(row["solution"]))
        if extracted is not None:
            return normalize_answer(extracted)
    raise ValueError("row has neither an 'answer' nor a boxed 'solution'")


__all__ = ["extract_answer", "extract_boxed", "normalize_answer", "reference_answer"]
