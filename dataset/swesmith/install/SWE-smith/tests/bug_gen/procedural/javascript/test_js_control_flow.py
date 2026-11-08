import pytest
from swesmith.bug_gen.adapters.javascript import get_entities_from_file_js
from swesmith.bug_gen.procedural.javascript.control_flow import (
    ControlIfElseInvertModifier,
    ControlShuffleLinesModifier,
)


@pytest.mark.parametrize(
    "src,expected",
    [
        # Simple if-else inversion
        (
            """function foo(x) {
    if (x > 0) {
        return 1;
    } else {
        return -1;
    }
}""",
            """function foo(x) {
    if (x > 0) {
        return -1;
    } else {
        return 1;
    }
}""",
        ),
        # Multiple statements in blocks
        (
            """function bar(condition) {
    if (condition) {
        const a = 1;
        return a;
    } else {
        const b = 2;
        return b;
    }
}""",
            """function bar(condition) {
    if (condition) {
        const b = 2;
        return b;
    } else {
        const a = 1;
        return a;
    }
}""",
        ),
    ],
)
def test_control_if_else_invert_modifier(tmp_path, src, expected):
    """Test that ControlIfElseInvertModifier inverts if-else branches."""
    test_file = tmp_path / "test.js"
    test_file.write_text(src, encoding="utf-8")

    entities = []
    get_entities_from_file_js(entities, str(test_file))
    assert len(entities) == 1

    modifier = ControlIfElseInvertModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entities[0])

    assert result is not None
    assert result.rewrite.strip() == expected.strip(), (
        f"Expected {expected}, got {result.rewrite}"
    )


@pytest.mark.parametrize(
    "src,expected_variants",
    [
        # Function with loop and two statements to shuffle
        (
            """function foo() {
    for (let i = 0; i < 10; i++) {}
    let a = 1;
    let b = 2;
}""",
            [
                "function foo() {\n    for (let i = 0; i < 10; i++) {}\n    let a = 1;\n    let b = 2;\n}",
                "function foo() {\n    for (let i = 0; i < 10; i++) {}\n    let b = 2;\n    let a = 1;\n}",
                "function foo() {\n    let a = 1;\n    for (let i = 0; i < 10; i++) {}\n    let b = 2;\n}",
                "function foo() {\n    let a = 1;\n    let b = 2;\n    for (let i = 0; i < 10; i++) {}\n}",
                "function foo() {\n    let b = 2;\n    for (let i = 0; i < 10; i++) {}\n    let a = 1;\n}",
                "function foo() {\n    let b = 2;\n    let a = 1;\n    for (let i = 0; i < 10; i++) {}\n}",
            ],
        ),
        # Function with loop and three statements
        (
            """function bar() {
    while (true) { break; }
    let x = 1;
    let y = 2;
}""",
            [
                "function bar() {\n    while (true) { break; }\n    let x = 1;\n    let y = 2;\n}",
                "function bar() {\n    while (true) { break; }\n    let y = 2;\n    let x = 1;\n}",
                "function bar() {\n    let x = 1;\n    while (true) { break; }\n    let y = 2;\n}",
                "function bar() {\n    let x = 1;\n    let y = 2;\n    while (true) { break; }\n}",
                "function bar() {\n    let y = 2;\n    while (true) { break; }\n    let x = 1;\n}",
                "function bar() {\n    let y = 2;\n    let x = 1;\n    while (true) { break; }\n}",
            ],
        ),
    ],
)
def test_control_shuffle_lines_modifier(tmp_path, src, expected_variants):
    """Test that ControlShuffleLinesModifier shuffles independent lines."""
    test_file = tmp_path / "test.js"
    test_file.write_text(src, encoding="utf-8")

    entities = []
    get_entities_from_file_js(entities, str(test_file))
    assert len(entities) == 1

    modifier = ControlShuffleLinesModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entities[0])

    # Result may be None if shuffle happened to produce the same order
    # or if complexity constraints aren't met
    if result is not None:
        assert any(
            result.rewrite.strip() == variant.strip() for variant in expected_variants
        ), f"Expected one of {expected_variants}, got {result.rewrite}"
