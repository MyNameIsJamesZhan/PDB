import pytest
from swesmith.bug_gen.adapters.javascript import get_entities_from_file_js
from swesmith.bug_gen.procedural.javascript.operations import (
    OperationChangeModifier,
    OperationFlipOperatorModifier,
    OperationSwapOperandsModifier,
    OperationChangeConstantsModifier,
    OperationBreakChainsModifier,
    AugmentedAssignmentSwapModifier,
    TernaryOperatorSwapModifier,
    FunctionArgumentSwapModifier,
)


@pytest.mark.parametrize(
    "src,expected_variants",
    [
        # Addition can change to subtraction
        (
            """function foo(a, b) {
    return a + b;
}""",
            [
                "function foo(a, b) {\n    return a - b;\n}",
            ],
        ),
        # Multiplication can change to division or modulo
        (
            """function bar(x, y) {
    return x * y;
}""",
            [
                "function bar(x, y) {\n    return x / y;\n}",
                "function bar(x, y) {\n    return x % y;\n}",
            ],
        ),
        # Bitwise operators
        (
            """function baz(a, b) {
    return a & b;
}""",
            [
                "function baz(a, b) {\n    return a | b;\n}",
                "function baz(a, b) {\n    return a ^ b;\n}",
            ],
        ),
    ],
)
def test_operation_change_modifier(tmp_path, src, expected_variants):
    """Test that OperationChangeModifier changes operators within the same category."""
    test_file = tmp_path / "test.js"
    test_file.write_text(src, encoding="utf-8")

    entities = []
    get_entities_from_file_js(entities, str(test_file))
    assert len(entities) == 1

    modifier = OperationChangeModifier(likelihood=1.0, seed=42)

    found_variant = False
    for _ in range(20):
        result = modifier.modify(entities[0])
        if (
            result
            and result.rewrite != src
            and any(
                result.rewrite.strip() == variant.strip()
                for variant in expected_variants
            )
        ):
            found_variant = True
            break

    assert found_variant, (
        f"Expected one of {expected_variants}, but got {result.rewrite if result else 'None'}"
    )


@pytest.mark.parametrize(
    "src,expected",
    [
        # Equality flips to inequality
        (
            """function foo(a, b) {
    return a === b;
}""",
            """function foo(a, b) {
    return a !== b;
}""",
        ),
        # Less than flips to greater than or equal
        (
            """function bar(x, y) {
    return x < y;
}""",
            """function bar(x, y) {
    return x >= y;
}""",
        ),
        # Logical AND flips to OR
        (
            """function baz(a, b) {
    return a && b;
}""",
            """function baz(a, b) {
    return a || b;
}""",
        ),
        # Addition flips to subtraction
        (
            """function qux(a, b) {
    return a + b;
}""",
            """function qux(a, b) {
    return a - b;
}""",
        ),
    ],
)
def test_operation_flip_operator_modifier(tmp_path, src, expected):
    """Test that OperationFlipOperatorModifier flips operators to their opposites."""
    test_file = tmp_path / "test.js"
    test_file.write_text(src, encoding="utf-8")

    entities = []
    get_entities_from_file_js(entities, str(test_file))
    assert len(entities) == 1

    modifier = OperationFlipOperatorModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entities[0])

    assert result is not None
    assert result.rewrite.strip() == expected.strip(), (
        f"Expected {expected}, got {result.rewrite}"
    )


@pytest.mark.parametrize(
    "src,expected",
    [
        # Swap operands in addition (commutative)
        (
            """function foo(a, b) {
    return a + b;
}""",
            """function foo(a, b) {
    return b + a;
}""",
        ),
        # Swap operands with comparison (flips operator too)
        (
            """function bar(x, y) {
    return x < y;
}""",
            """function bar(x, y) {
    return y > x;
}""",
        ),
        # Swap operands in subtraction
        (
            """function baz(a, b) {
    return a - b;
}""",
            """function baz(a, b) {
    return b - a;
}""",
        ),
    ],
)
def test_operation_swap_operands_modifier(tmp_path, src, expected):
    """Test that OperationSwapOperandsModifier swaps operands in binary expressions."""
    test_file = tmp_path / "test.js"
    test_file.write_text(src, encoding="utf-8")

    entities = []
    get_entities_from_file_js(entities, str(test_file))
    assert len(entities) == 1

    modifier = OperationSwapOperandsModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entities[0])

    assert result is not None
    assert result.rewrite.strip() == expected.strip(), (
        f"Expected {expected}, got {result.rewrite}"
    )


@pytest.mark.parametrize(
    "src,expected_variants",
    [
        # Change integer constant by +/- 1 or 2
        (
            """function foo() {
    return 2 + x;
}""",
            [
                "function foo() {\n    return 1 + x;\n}",
                "function foo() {\n    return 3 + x;\n}",
                "function foo() {\n    return 0 + x;\n}",
                "function foo() {\n    return 4 + x;\n}",
            ],
        ),
        # Change constant on right side
        (
            """function bar() {
    return y - 5;
}""",
            [
                "function bar() {\n    return y - 4;\n}",
                "function bar() {\n    return y - 6;\n}",
                "function bar() {\n    return y - 3;\n}",
                "function bar() {\n    return y - 7;\n}",
            ],
        ),
    ],
)
def test_operation_change_constants_modifier(tmp_path, src, expected_variants):
    """Test that OperationChangeConstantsModifier changes integer constants."""
    test_file = tmp_path / "test.js"
    test_file.write_text(src, encoding="utf-8")

    entities = []
    get_entities_from_file_js(entities, str(test_file))
    assert len(entities) == 1

    modifier = OperationChangeConstantsModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entities[0])

    assert result is not None
    assert any(
        result.rewrite.strip() == variant.strip() for variant in expected_variants
    ), f"Expected one of {expected_variants}, got {result.rewrite}"


@pytest.mark.parametrize(
    "src,expected_variants",
    [
        # Break addition chain by removing one operand
        (
            """function foo(a, b, c) {
    return a + b + c;
}""",
            [
                "function foo(a, b, c) {\n    return b + c;\n}",
                "function foo(a, b, c) {\n    return a + b;\n}",
            ],
        ),
        # Break multiplication chain (left-associative)
        (
            """function bar(x, y, z) {
    return x * y * z;
}""",
            [
                "function bar(x, y, z) {\n    return y * z;\n}",
                "function bar(x, y, z) {\n    return x * y;\n}",
            ],
        ),
    ],
)
def test_operation_break_chains_modifier(tmp_path, src, expected_variants):
    """Test that OperationBreakChainsModifier breaks operation chains."""
    test_file = tmp_path / "test.js"
    test_file.write_text(src, encoding="utf-8")

    entities = []
    get_entities_from_file_js(entities, str(test_file))
    assert len(entities) == 1

    modifier = OperationBreakChainsModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entities[0])

    assert result is not None
    assert any(
        result.rewrite.strip() == variant.strip() for variant in expected_variants
    ), f"Expected one of {expected_variants}, got {result.rewrite}"


@pytest.mark.parametrize(
    "src,expected_variants",
    [
        # += swaps to -=
        (
            """function foo(x) {
    x += 5;
    return x;
}""",
            [
                "function foo(x) {\n    x -= 5;\n    return x;\n}",
            ],
        ),
        # *= swaps to /=
        (
            """function bar(y) {
    y *= 2;
    return y;
}""",
            [
                "function bar(y) {\n    y /= 2;\n    return y;\n}",
            ],
        ),
        # ++ swaps to --
        (
            """function baz(n) {
    n++;
    return n;
}""",
            [
                "function baz(n) {\n    n--;\n    return n;\n}",
            ],
        ),
    ],
)
def test_augmented_assignment_swap_modifier(tmp_path, src, expected_variants):
    """Test that AugmentedAssignmentSwapModifier swaps augmented assignment operators."""
    test_file = tmp_path / "test.js"
    test_file.write_text(src, encoding="utf-8")

    entities = []
    get_entities_from_file_js(entities, str(test_file))
    assert len(entities) == 1

    modifier = AugmentedAssignmentSwapModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entities[0])

    assert result is not None
    assert any(
        result.rewrite.strip() == variant.strip() for variant in expected_variants
    ), f"Expected one of {expected_variants}, got {result.rewrite}"


@pytest.mark.parametrize(
    "src,expected_variants",
    [
        # Ternary branches swap
        (
            """function foo(condition) {
    return condition ? "yes" : "no";
}""",
            [
                'function foo(condition) {\n    return condition ? "no" : "yes";\n}',
                'function foo(condition) {\n    return !(condition) ? "yes" : "no";\n}',
            ],
        ),
        # Ternary with numbers
        (
            """function bar(x) {
    return x > 0 ? 1 : -1;
}""",
            [
                "function bar(x) {\n    return x > 0 ? -1 : 1;\n}",
                "function bar(x) {\n    return !(x > 0) ? 1 : -1;\n}",
            ],
        ),
    ],
)
def test_ternary_operator_swap_modifier(tmp_path, src, expected_variants):
    """Test that TernaryOperatorSwapModifier modifies ternary operators."""
    test_file = tmp_path / "test.js"
    test_file.write_text(src, encoding="utf-8")

    entities = []
    get_entities_from_file_js(entities, str(test_file))
    assert len(entities) == 1

    modifier = TernaryOperatorSwapModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entities[0])

    assert result is not None
    assert any(
        result.rewrite.strip() == variant.strip() for variant in expected_variants
    ), f"Expected one of {expected_variants}, got {result.rewrite}"


@pytest.mark.parametrize(
    "src,expected_variants",
    [
        # Swap two arguments
        (
            """function foo() {
    return add(1, 2);
}""",
            [
                "function foo() {\n    return add(2, 1);\n}",
            ],
        ),
        # Swap in function with three arguments
        (
            """function bar() {
    return compute(a, b, c);
}""",
            [
                "function bar() {\n    return compute(b, a, c);\n}",
                "function bar() {\n    return compute(a, c, b);\n}",
            ],
        ),
    ],
)
def test_function_argument_swap_modifier(tmp_path, src, expected_variants):
    """Test that FunctionArgumentSwapModifier swaps adjacent arguments."""
    test_file = tmp_path / "test.js"
    test_file.write_text(src, encoding="utf-8")

    entities = []
    get_entities_from_file_js(entities, str(test_file))
    assert len(entities) == 1

    modifier = FunctionArgumentSwapModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entities[0])

    assert result is not None
    assert any(
        result.rewrite.strip() == variant.strip() for variant in expected_variants
    ), f"Expected one of {expected_variants}, got {result.rewrite}"
