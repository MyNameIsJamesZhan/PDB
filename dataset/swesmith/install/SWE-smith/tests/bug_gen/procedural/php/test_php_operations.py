import pytest
from swesmith.bug_gen.adapters.php import get_entities_from_file_php
from swesmith.bug_gen.procedural.php.operations import (
    _safe_decode,
    OperationChangeModifier,
    OperationFlipOperatorModifier,
    OperationSwapOperandsModifier,
    OperationChangeConstantsModifier,
    OperationBreakChainsModifier,
    AugmentedAssignmentSwapModifier,
    TernaryOperatorSwapModifier,
    FunctionArgumentSwapModifier,
)

PHP_PREFIX = "<?php\n"


def _get_entity(tmp_path, src):
    test_file = tmp_path / "test.php"
    test_file.write_text(PHP_PREFIX + src, encoding="utf-8")
    entities = []
    get_entities_from_file_php(entities, str(test_file))
    assert len(entities) >= 1
    return entities[0]


@pytest.mark.parametrize(
    "src,expected_variants",
    [
        (
            """function foo($a, $b) {
    return $a + $b;
}""",
            ["return $a - $b;"],
        ),
        (
            """function bar($x, $y) {
    return $x * $y;
}""",
            ["return $x / $y;", "return $x % $y;"],
        ),
        (
            """function baz($a, $b) {
    return $a & $b;
}""",
            ["return $a | $b;", "return $a ^ $b;"],
        ),
        # PHP string concatenation
        (
            """function concat($a, $b) {
    return $a . $b;
}""",
            ["return $a + $b;"],
        ),
    ],
)
def test_operation_change_modifier(tmp_path, src, expected_variants):
    entity = _get_entity(tmp_path, src)
    modifier = OperationChangeModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)

    assert result is not None
    assert any(v in result.rewrite for v in expected_variants), (
        f"Expected one of {expected_variants} in output, got: {result.rewrite}"
    )


@pytest.mark.parametrize(
    "src,expected_substring",
    [
        (
            """function foo($a, $b) {
    return $a === $b;
}""",
            "return $a !== $b;",
        ),
        (
            """function foo($a, $b) {
    return $a !== $b;
}""",
            "return $a === $b;",
        ),
        (
            """function foo($a, $b) {
    return $a == $b;
}""",
            "return $a != $b;",
        ),
        (
            """function foo($a, $b) {
    return $a != $b;
}""",
            "return $a == $b;",
        ),
        (
            """function foo($x, $y) {
    return $x < $y;
}""",
            "return $x >= $y;",
        ),
        (
            """function foo($x, $y) {
    return $x >= $y;
}""",
            "return $x < $y;",
        ),
        (
            """function foo($x, $y) {
    return $x > $y;
}""",
            "return $x <= $y;",
        ),
        (
            """function foo($x, $y) {
    return $x <= $y;
}""",
            "return $x > $y;",
        ),
        (
            """function foo($a, $b) {
    return $a && $b;
}""",
            "return $a || $b;",
        ),
        (
            """function foo($a, $b) {
    return $a || $b;
}""",
            "return $a && $b;",
        ),
        (
            """function foo($a, $b) {
    return $a and $b;
}""",
            "return $a or $b;",
        ),
        (
            """function foo($a, $b) {
    return $a or $b;
}""",
            "return $a and $b;",
        ),
    ],
)
def test_operation_flip_operator_modifier(tmp_path, src, expected_substring):
    entity = _get_entity(tmp_path, src)
    modifier = OperationFlipOperatorModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)

    assert result is not None
    assert expected_substring in result.rewrite


@pytest.mark.parametrize(
    "src,expected_substring",
    [
        (
            """function foo($a, $b) {
    return $a + $b;
}""",
            "return $b + $a;",
        ),
        (
            """function bar($x, $y) {
    return $x - $y;
}""",
            "return $y - $x;",
        ),
    ],
)
def test_operation_swap_operands_modifier(tmp_path, src, expected_substring):
    entity = _get_entity(tmp_path, src)
    modifier = OperationSwapOperandsModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)

    assert result is not None
    assert expected_substring in result.rewrite


@pytest.mark.parametrize(
    "src,original,expected_variants",
    [
        (
            """function foo() {
    return 2 + $x;
}""",
            "return 2 + $x;",
            ["return 1 + $x;", "return 3 + $x;", "return 0 + $x;", "return 4 + $x;"],
        ),
        (
            """function bar() {
    return $y - 5;
}""",
            "return $y - 5;",
            ["return $y - 4;", "return $y - 6;", "return $y - 3;", "return $y - 7;"],
        ),
    ],
)
def test_operation_change_constants_modifier(
    tmp_path, src, original, expected_variants
):
    entity = _get_entity(tmp_path, src)
    modifier = OperationChangeConstantsModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)

    assert result is not None
    assert original not in result.rewrite
    assert any(v in result.rewrite for v in expected_variants), (
        f"Expected one of {expected_variants} in output, got: {result.rewrite}"
    )


@pytest.mark.parametrize(
    "src,dropped_operand",
    [
        (
            """function foo($a, $b, $c) {
    return $a + $b + $c;
}""",
            "$a",
        ),
        (
            """function bar($x, $y, $z) {
    return $x * $y * $z;
}""",
            "$x",
        ),
    ],
)
def test_operation_break_chains_modifier(tmp_path, src, dropped_operand):
    entity = _get_entity(tmp_path, src)
    modifier = OperationBreakChainsModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)

    assert result is not None
    assert dropped_operand + " " not in result.rewrite.split("return")[1]


@pytest.mark.parametrize(
    "src,expected_substring",
    [
        (
            """function foo($x) {
    $x += 5;
    return $x;
}""",
            "$x -= 5;",
        ),
        (
            """function bar($y) {
    $y *= 2;
    return $y;
}""",
            "$y /= 2;",
        ),
        (
            """function baz($n) {
    $n++;
    return $n;
}""",
            "$n--;",
        ),
    ],
)
def test_augmented_assignment_swap_modifier(tmp_path, src, expected_substring):
    entity = _get_entity(tmp_path, src)
    modifier = AugmentedAssignmentSwapModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)

    assert result is not None
    assert expected_substring in result.rewrite


@pytest.mark.parametrize(
    "src,expected_substring",
    [
        (
            """function foo($condition) {
    return $condition ? "yes" : "no";
}""",
            'return $condition ? "no" : "yes";',
        ),
        (
            """function bar($x) {
    return $x > 0 ? 1 : -1;
}""",
            "return $x > 0 ? -1 : 1;",
        ),
    ],
)
def test_ternary_operator_swap_modifier(tmp_path, src, expected_substring):
    entity = _get_entity(tmp_path, src)
    modifier = TernaryOperatorSwapModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)

    assert result is not None
    assert expected_substring in result.rewrite


@pytest.mark.parametrize(
    "src,expected_call",
    [
        (
            """function foo() {
    return add(1, 2);
}""",
            "add(2, 1)",
        ),
        (
            """function bar() {
    return compute($a, $b, $c);
}""",
            None,
        ),
        (
            """function baz($obj) {
    return $obj->method(1, 2);
}""",
            "method(2, 1)",
        ),
        (
            """function qux() {
    return MyClass::create($x, $y);
}""",
            "create($y, $x)",
        ),
    ],
)
def test_function_argument_swap_modifier(tmp_path, src, expected_call):
    entity = _get_entity(tmp_path, src)
    modifier = FunctionArgumentSwapModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)

    assert result is not None
    assert result.rewrite != src
    if expected_call:
        assert expected_call in result.rewrite
    else:
        assert "compute(" in result.rewrite


def test_safe_decode_valid():
    assert _safe_decode(b"hello") == "hello"


def test_safe_decode_invalid_utf8():
    result = _safe_decode(b"\xff\xfe", fallback="fallback")
    assert result == "fallback"


def test_safe_decode_invalid_utf8_default_fallback():
    result = _safe_decode(b"\xff\xfe")
    assert result == ""


def test_operation_change_no_changes_likelihood_zero(tmp_path):
    src = """function foo($a, $b) {
    return $a + $b;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = OperationChangeModifier(likelihood=0.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_operation_change_no_binary_ops(tmp_path):
    src = """function foo($x) {
    return $x;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = OperationChangeModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_operation_flip_no_changes_likelihood_zero(tmp_path):
    src = """function foo($a, $b) {
    return $a === $b;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = OperationFlipOperatorModifier(likelihood=0.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_operation_swap_operands_no_changes_likelihood_zero(tmp_path):
    src = """function foo($a, $b) {
    return $a + $b;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = OperationSwapOperandsModifier(likelihood=0.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_operation_swap_operands_comparison_no_flip(tmp_path):
    """Test that comparison operators are NOT flipped when operands are swapped (to actually change semantics)."""
    src = """function foo($a, $b) {
    return $a < $b;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = OperationSwapOperandsModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)
    assert result is not None
    # Operator should stay as "<" (not flipped to ">"), so "$b < $a" changes behavior
    assert "$b < $a" in result.rewrite


def test_operation_change_constants_no_integers(tmp_path):
    src = """function foo($x) {
    return $x;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = OperationChangeConstantsModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_operation_change_constants_likelihood_zero(tmp_path):
    src = """function foo() {
    return 42;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = OperationChangeConstantsModifier(likelihood=0.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_operation_break_chains_no_chains(tmp_path):
    src = """function foo($a, $b) {
    return $a + $b;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = OperationBreakChainsModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_operation_break_chains_right_chain(tmp_path):
    """Test breaking chains where the right side is a binary expression."""
    src = """function foo($a, $b, $c) {
    return $a + ($b + $c);
}"""
    entity = _get_entity(tmp_path, src)
    modifier = OperationBreakChainsModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)
    # Parenthesized expression isn't detected as a chain, so no modification
    assert result is None


def test_operation_break_chains_likelihood_zero(tmp_path):
    src = """function foo($a, $b, $c) {
    return $a + $b + $c;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = OperationBreakChainsModifier(likelihood=0.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_augmented_assignment_swap_no_assignments(tmp_path):
    src = """function foo($x) {
    return $x;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = AugmentedAssignmentSwapModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_augmented_assignment_swap_likelihood_zero(tmp_path):
    src = """function foo($x) {
    $x += 5;
    return $x;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = AugmentedAssignmentSwapModifier(likelihood=0.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_ternary_swap_negate_condition(tmp_path):
    """Test that the negate_condition path is exercised."""
    src = """function foo($x) {
    return $x > 0 ? "yes" : "no";
}"""
    entity = _get_entity(tmp_path, src)
    # seed=0 produces negation
    modifier = TernaryOperatorSwapModifier(likelihood=1.0, seed=0)
    result = modifier.modify(entity)
    assert result is not None
    assert '!($x > 0) ? "yes" : "no"' in result.rewrite


def test_ternary_swap_no_ternary(tmp_path):
    src = """function foo($x) {
    return $x;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = TernaryOperatorSwapModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_ternary_swap_likelihood_zero(tmp_path):
    src = """function foo($x) {
    return $x > 0 ? "yes" : "no";
}"""
    entity = _get_entity(tmp_path, src)
    modifier = TernaryOperatorSwapModifier(likelihood=0.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_function_argument_swap_no_calls(tmp_path):
    src = """function foo($x) {
    return $x;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = FunctionArgumentSwapModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_function_argument_swap_single_arg(tmp_path):
    src = """function foo() {
    return strlen("hello");
}"""
    entity = _get_entity(tmp_path, src)
    modifier = FunctionArgumentSwapModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_function_argument_swap_likelihood_zero(tmp_path):
    src = """function foo() {
    return add(1, 2);
}"""
    entity = _get_entity(tmp_path, src)
    modifier = FunctionArgumentSwapModifier(likelihood=0.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_operation_change_exponentiation(tmp_path):
    """Test that ** operator can be swapped with *, /, %."""
    src = """function foo($a, $b) {
    return $a ** $b;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = OperationChangeModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)
    assert result is not None
    assert any(
        op in result.rewrite
        for op in ["return $a * $b;", "return $a / $b;", "return $a % $b;"]
    ), f"Expected ** to be swapped, got: {result.rewrite}"


def test_change_constants_float(tmp_path):
    """Test that float literals are modified."""
    src = """function foo() {
    return 3.14 + $x;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = OperationChangeConstantsModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)
    assert result is not None
    assert "3.14" not in result.rewrite


def test_change_constants_hex(tmp_path):
    """Test that hex literals are modified."""
    src = """function foo() {
    return 0xFF + $x;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = OperationChangeConstantsModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)
    assert result is not None
    assert "0xFF" not in result.rewrite


def test_change_constants_octal(tmp_path):
    """Test that octal literals are handled correctly."""
    src = """function foo() {
    return 077 + $x;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = OperationChangeConstantsModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)
    assert result is not None
    assert "077" not in result.rewrite


def test_break_chains_nested_no_corruption(tmp_path):
    """Test that nested expressions don't get corrupted by overlapping changes."""
    src = """function foo($a, $b, $c, $d, $extra) {
    return foo($a + $b + $c + $d) + $extra;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = OperationBreakChainsModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)
    assert result is not None
    assert "function foo" in result.rewrite
    assert "return" in result.rewrite
