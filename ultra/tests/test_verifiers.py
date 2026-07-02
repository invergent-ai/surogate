from ultra.grading.verifiers import math_equal


def test_math_equal_accepts_common_fraction_formats():
    assert math_equal("`4/7`", "4/7") == 1.0
    assert math_equal("The answer is **4/7**.", "4/7") == 1.0
    assert math_equal(r"\boxed{\dfrac{4}{7}}", "4/7") == 1.0
    assert math_equal(r"Therefore, \frac{4}{7}.", "4/7") == 1.0
    assert math_equal("4/8", "4/7") == 0.0
