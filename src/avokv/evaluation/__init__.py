from .answers import extract_answer, extract_boxed, normalize_answer, reference_answer


def math_equal(*args, **kwargs):
    """Load the optional symbolic evaluator only when evaluation is requested."""
    from .grader import math_equal as _math_equal

    return _math_equal(*args, **kwargs)


__all__ = [
    "extract_answer",
    "extract_boxed",
    "math_equal",
    "normalize_answer",
    "reference_answer",
]
