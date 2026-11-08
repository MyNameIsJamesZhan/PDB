import pytest
from swesmith.bug_gen.adapters.php import get_entities_from_file_php
from swesmith.bug_gen.procedural.php.control_flow import (
    ControlIfElseInvertModifier,
    ControlShuffleLinesModifier,
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
    "src,orig_if_body,orig_else_body",
    [
        (
            """function foo($x) {
    if ($x > 0) {
        return "positive";
    } else {
        return "non-positive";
    }
}""",
            'return "positive";',
            'return "non-positive";',
        ),
        (
            """function bar($a, $b) {
    if ($a === $b) {
        $result = true;
    } else {
        $result = false;
    }
    return $result;
}""",
            "$result = true;",
            "$result = false;",
        ),
    ],
)
def test_control_if_else_invert_modifier(tmp_path, src, orig_if_body, orig_else_body):
    entity = _get_entity(tmp_path, src)
    modifier = ControlIfElseInvertModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)

    assert result is not None
    assert result.rewrite != src
    rewrite = result.rewrite
    if_idx = rewrite.index("if")
    else_idx = rewrite.index("else", if_idx)
    if_block = rewrite[if_idx:else_idx]
    else_block = rewrite[else_idx:]
    assert orig_else_body in if_block
    assert orig_if_body in else_block


@pytest.mark.parametrize(
    "src",
    [
        """function foo($x) {
    $a = 1;
    $b = 2;
    $c = 3;
    for ($i = 0; $i < $x; $i++) {
        echo $i;
    }
    return $a + $b + $c;
}""",
    ],
)
def test_control_shuffle_lines_modifier(tmp_path, src):
    entity = _get_entity(tmp_path, src)
    modifier = ControlShuffleLinesModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)

    assert result is not None
    assert result.rewrite != src
    for stmt in ["$a = 1;", "$b = 2;", "$c = 3;"]:
        assert stmt in result.rewrite


def test_control_shuffle_lines_method(tmp_path):
    """Shuffle lines within a class method."""
    src = """class Calc {
    public function compute($x) {
        $a = 1;
        $b = 2;
        $c = 3;
        return $a + $b + $c;
    }
}"""
    test_file = tmp_path / "test.php"
    test_file.write_text(PHP_PREFIX + src, encoding="utf-8")
    entities = []
    get_entities_from_file_php(entities, str(test_file))
    method_entities = [e for e in entities if e.node.type == "method_declaration"]
    assert len(method_entities) >= 1

    modifier = ControlShuffleLinesModifier(likelihood=1.0, seed=42)
    result = modifier.modify(method_entities[0])

    assert result is not None
    assert result.rewrite != method_entities[0].src_code
    for stmt in ["$a = 1;", "$b = 2;", "$c = 3;"]:
        assert stmt in result.rewrite


def test_control_if_else_invert_no_else(tmp_path):
    """If-only (no else) should return None."""
    src = """function foo($x) {
    if ($x > 0) {
        return "positive";
    }
    return "other";
}"""
    entity = _get_entity(tmp_path, src)
    modifier = ControlIfElseInvertModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_control_if_else_invert_likelihood_zero(tmp_path):
    """All retries fail when likelihood=0."""
    src = """function foo($x) {
    if ($x > 0) {
        return "positive";
    } else {
        return "non-positive";
    }
}"""
    entity = _get_entity(tmp_path, src)
    modifier = ControlIfElseInvertModifier(likelihood=0.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_control_shuffle_lines_no_function_body(tmp_path):
    """Code without a proper function body should return None."""
    src = """function foo($x) {
    return $x;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = ControlShuffleLinesModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)
    # Only one statement, can't shuffle
    assert result is None


def test_control_shuffle_lines_likelihood_zero(tmp_path):
    src = """function foo($x) {
    $a = 1;
    $b = 2;
    $c = 3;
    return $a + $b + $c;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = ControlShuffleLinesModifier(likelihood=0.0, seed=42)
    result = modifier.modify(entity)
    assert result is None
