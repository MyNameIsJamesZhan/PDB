import pytest
from swesmith.bug_gen.adapters.javascript import get_entities_from_file_js
from swesmith.bug_gen.procedural.javascript.remove import (
    RemoveLoopModifier,
    RemoveConditionalModifier,
    RemoveAssignmentModifier,
    RemoveTernaryModifier,
)


def normalize_code(code: str) -> str:
    """Normalize code for comparison by collapsing consecutive whitespace/newlines."""
    # Remove leading/trailing whitespace from each line, filter empty lines
    lines = [line.strip() for line in code.strip().split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


@pytest.mark.parametrize(
    "src,expected",
    [
        # Remove for loop
        (
            """function foo() {
    for (let i = 0; i < 3; i++) {
        console.log(i);
    }
    return 1;
}""",
            """function foo() {
    
    return 1;
}""",
        ),
        # Remove while loop
        (
            """function bar() {
    while (true) {
        break;
    }
    return 2;
}""",
            """function bar() {
    
    return 2;
}""",
        ),
        # Remove for-of loop
        (
            """function baz(arr) {
    let sum = 0;
    for (const item of arr) {
        sum += item;
    }
    return sum;
}""",
            """function baz(arr) {
    let sum = 0;
    
    return sum;
}""",
        ),
        # Remove for-in loop
        (
            """function qux(obj) {
    let keys = [];
    for (const key in obj) {
        keys.push(key);
    }
    return keys;
}""",
            """function qux(obj) {
    let keys = [];
    
    return keys;
}""",
        ),
    ],
)
def test_remove_loop_modifier(tmp_path, src, expected):
    """Test that RemoveLoopModifier removes loop statements."""
    test_file = tmp_path / "test.js"
    test_file.write_text(src, encoding="utf-8")

    entities = []
    get_entities_from_file_js(entities, str(test_file))
    assert len(entities) == 1

    modifier = RemoveLoopModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entities[0])

    assert result is not None
    assert result.rewrite.strip() == expected.strip(), (
        f"Expected {expected}, got {result.rewrite}"
    )


@pytest.mark.parametrize(
    "src,expected",
    [
        # Remove if statement (no else)
        (
            """function foo(x) {
    if (x > 0) {
        return x;
    }
    return 0;
}""",
            """function foo(x) {
    
    return 0;
}""",
        ),
        # Remove complete if-else
        (
            """function bar(x) {
    if (x < 0) {
        return -1;
    } else {
        return 1;
    }
}""",
            """function bar(x) {
    
}""",
        ),
        # Remove if with multiple statements
        (
            """function baz(x) {
    let result = 0;
    if (x > 10) {
        result = x * 2;
    }
    return result;
}""",
            """function baz(x) {
    let result = 0;
    
    return result;
}""",
        ),
    ],
)
def test_remove_conditional_modifier(tmp_path, src, expected):
    """Test that RemoveConditionalModifier removes conditional statements."""
    test_file = tmp_path / "test.js"
    test_file.write_text(src, encoding="utf-8")

    entities = []
    get_entities_from_file_js(entities, str(test_file))
    assert len(entities) == 1

    modifier = RemoveConditionalModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entities[0])

    assert result is not None
    assert result.rewrite.strip() == expected.strip(), (
        f"Expected {expected}, got {result.rewrite}"
    )


@pytest.mark.parametrize(
    "src,expected",
    [
        # Remove variable declaration with var
        (
            """function foo() {
    var x = 1;
    return x;
}""",
            """function foo() {
    return x;
}""",
        ),
        # Remove var declaration with expression
        (
            """function bar() {
    var y = 2 + 3;
    return y;
}""",
            """function bar() {
    return y;
}""",
        ),
        # With likelihood=1.0, ALL assignments are removed (both var and +=)
        (
            """function baz() {
    var z = 0;
    z += 5;
    return z;
}""",
            """function baz() {
    return z;
}""",
        ),
    ],
)
def test_remove_assignment_modifier(tmp_path, src, expected):
    """Test that RemoveAssignmentModifier removes assignment statements."""
    test_file = tmp_path / "test.js"
    test_file.write_text(src, encoding="utf-8")

    entities = []
    get_entities_from_file_js(entities, str(test_file))
    assert len(entities) == 1

    modifier = RemoveAssignmentModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entities[0])

    assert result is not None
    # Use normalize_code to handle whitespace variations in the output
    assert normalize_code(result.rewrite) == normalize_code(expected), (
        f"Expected {expected}, got {result.rewrite}"
    )


@pytest.mark.parametrize(
    "src,expected_variants",
    [
        # Remove ternary - keep consequent
        (
            """function foo(condition) {
    return condition ? "yes" : "no";
}""",
            [
                'function foo(condition) {\n    return "yes";\n}',
                'function foo(condition) {\n    return "no";\n}',
            ],
        ),
        # Remove ternary with numbers
        (
            """function bar(x) {
    return x > 0 ? 1 : -1;
}""",
            [
                "function bar(x) {\n    return 1;\n}",
                "function bar(x) {\n    return -1;\n}",
            ],
        ),
        # Remove ternary with expressions
        (
            """function baz(a, b) {
    return a > b ? a : b;
}""",
            [
                "function baz(a, b) {\n    return a;\n}",
                "function baz(a, b) {\n    return b;\n}",
            ],
        ),
    ],
)
def test_remove_ternary_modifier(tmp_path, src, expected_variants):
    """Test that RemoveTernaryModifier removes ternary expressions by keeping one branch."""
    test_file = tmp_path / "test.js"
    test_file.write_text(src, encoding="utf-8")

    entities = []
    get_entities_from_file_js(entities, str(test_file))
    assert len(entities) == 1

    modifier = RemoveTernaryModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entities[0])

    assert result is not None
    assert any(
        result.rewrite.strip() == variant.strip() for variant in expected_variants
    ), f"Expected one of {expected_variants}, got {result.rewrite}"
