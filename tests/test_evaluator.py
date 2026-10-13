from __future__ import annotations

import pytest

from avokv.evaluation import math_equal


@pytest.mark.parametrize(
    ("prediction", "reference"),
    [
        ("42", "42"),
        ("0.5", r"\frac{1}{2}"),
        (r"p-q", r"p - q"),
    ],
)
def test_math_equivalence_regressions(prediction: str, reference: str) -> None:
    pytest.importorskip("latex2sympy2")
    assert math_equal(prediction, reference, timeout=False)
