from __future__ import annotations

from avokv.evaluation.answers import (
    extract_answer,
    extract_boxed,
    normalize_answer,
    reference_answer,
)


def test_extract_boxed_handles_nested_latex() -> None:
    text = r"Work here. Final answer: \boxed{\frac{3}{\sqrt{2}}}."
    assert extract_boxed(text) == r"\frac{3}{\sqrt{2}}"


def test_extract_answer_prefers_last_box() -> None:
    text = r"First \boxed{1}, correction \boxed{2}."
    assert extract_answer(text) == "2"


def test_extract_answer_falls_back_to_final_answer_marker() -> None:
    assert extract_answer("Therefore, Final answer: 42\n") == "42"


def test_reference_uses_answer_field_without_benchmark_specific_schema() -> None:
    assert reference_answer({"answer": r"\left( 3, \frac{\pi}{2} \right)"}) == (
        r"(3,\frac{\pi}{2})"
    )


def test_normalize_answer_removes_common_latex_wrappers() -> None:
    assert normalize_answer(r"$\mathrm{12}\%$") == "12"
