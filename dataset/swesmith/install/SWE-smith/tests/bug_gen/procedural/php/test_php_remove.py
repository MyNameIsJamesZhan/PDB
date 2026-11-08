import pytest
from swesmith.bug_gen.adapters.php import get_entities_from_file_php
from swesmith.bug_gen.procedural.php.remove import (
    RemoveLoopModifier,
    RemoveConditionalModifier,
    RemoveAssignmentModifier,
    RemoveTernaryModifier,
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
    "src,loop_keyword",
    [
        (
            """function foo($arr) {
    $sum = 0;
    for ($i = 0; $i < count($arr); $i++) {
        $sum += $arr[$i];
    }
    return $sum;
}""",
            "for",
        ),
        (
            """function bar($items) {
    $result = [];
    foreach ($items as $item) {
        $result[] = $item * 2;
    }
    return $result;
}""",
            "foreach",
        ),
        (
            """function baz($n) {
    $i = 0;
    while ($i < $n) {
        echo $i;
        $i++;
    }
}""",
            "while",
        ),
    ],
)
def test_remove_loop_modifier(tmp_path, src, loop_keyword):
    entity = _get_entity(tmp_path, src)
    modifier = RemoveLoopModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)

    assert result is not None
    assert result.rewrite != src
    assert loop_keyword not in result.rewrite


@pytest.mark.parametrize(
    "src",
    [
        """function foo($x) {
    if ($x > 0) {
        return "positive";
    }
    return "non-positive";
}""",
        """function bar($a, $b) {
    if ($a === $b) {
        return true;
    }
    return false;
}""",
    ],
)
def test_remove_conditional_modifier(tmp_path, src):
    entity = _get_entity(tmp_path, src)
    modifier = RemoveConditionalModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)

    assert result is not None
    assert result.rewrite != src
    assert len(result.rewrite) < len(src)


@pytest.mark.parametrize(
    "src,removed_statement",
    [
        (
            """function foo($x) {
    $y = $x + 1;
    return $y;
}""",
            "$y = $x + 1;",
        ),
        (
            """function bar($a) {
    $a += 5;
    return $a;
}""",
            "$a += 5;",
        ),
    ],
)
def test_remove_assignment_modifier(tmp_path, src, removed_statement):
    entity = _get_entity(tmp_path, src)
    modifier = RemoveAssignmentModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)

    assert result is not None
    assert removed_statement not in result.rewrite


@pytest.mark.parametrize(
    "src",
    [
        """function foo($x) {
    return $x > 0 ? "yes" : "no";
}""",
        """function bar($a, $b) {
    return $a === $b ? 1 : 0;
}""",
    ],
)
def test_remove_ternary_modifier(tmp_path, src):
    entity = _get_entity(tmp_path, src)
    modifier = RemoveTernaryModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)

    assert result is not None
    assert result.rewrite != src
    assert "?" not in result.rewrite


def test_remove_loop_no_loops(tmp_path):
    src = """function foo($x) {
    return $x;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = RemoveLoopModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_remove_loop_likelihood_zero(tmp_path):
    src = """function foo($arr) {
    $sum = 0;
    for ($i = 0; $i < count($arr); $i++) {
        $sum += $arr[$i];
    }
    return $sum;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = RemoveLoopModifier(likelihood=0.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_remove_conditional_no_conditionals(tmp_path):
    src = """function foo($x) {
    return $x;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = RemoveConditionalModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_remove_conditional_likelihood_zero(tmp_path):
    src = """function foo($x) {
    if ($x > 0) {
        return "positive";
    }
    return "other";
}"""
    entity = _get_entity(tmp_path, src)
    modifier = RemoveConditionalModifier(likelihood=0.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_remove_assignment_no_assignments(tmp_path):
    src = """function foo($x) {
    return $x;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = RemoveAssignmentModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_remove_assignment_likelihood_zero(tmp_path):
    src = """function foo($x) {
    $y = $x + 1;
    return $y;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = RemoveAssignmentModifier(likelihood=0.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_remove_ternary_no_ternary(tmp_path):
    src = """function foo($x) {
    return $x;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = RemoveTernaryModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_remove_ternary_likelihood_zero(tmp_path):
    src = """function foo($x) {
    return $x > 0 ? "yes" : "no";
}"""
    entity = _get_entity(tmp_path, src)
    modifier = RemoveTernaryModifier(likelihood=0.0, seed=42)
    result = modifier.modify(entity)
    assert result is None


def test_remove_assignment_nested_no_corruption(tmp_path):
    """Test that nested assignments like $a = $b = 5 don't corrupt output."""
    src = """function foo($x) {
    $a = $b = 5;
    echo $a + $b;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = RemoveAssignmentModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)
    assert result is not None
    # Should cleanly remove the whole statement, not corrupt 'echo' into 'o'
    assert "echo" in result.rewrite and "$a = $b = 5" not in result.rewrite


def test_remove_assignment_reference(tmp_path):
    """Test that reference assignments ($b = &$a) are removed."""
    src = """function foo($a) {
    $b = &$a;
    return $b;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = RemoveAssignmentModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)
    assert result is not None
    assert "&$a" not in result.rewrite


def test_remove_ternary_nested_no_corruption(tmp_path):
    """Test that nested ternaries don't lose trailing code."""
    src = """function foo($a, $b, $c, $d, $e) {
    return $a ? $b : $c ? $d : $e;
}"""
    entity = _get_entity(tmp_path, src)
    modifier = RemoveTernaryModifier(likelihood=1.0, seed=42)
    result = modifier.modify(entity)
    assert result is not None
    assert result.rewrite.rstrip().endswith("}")


def test_remove_ternary_elvis_operator(tmp_path):
    """Test that the elvis operator ($x ?: $default) is supported."""
    src = """function foo($x, $default) {
    return $x ?: $default;
}"""
    entity = _get_entity(tmp_path, src)
    found = False
    for seed in range(20):
        modifier = RemoveTernaryModifier(likelihood=1.0, seed=seed)
        result = modifier.modify(entity)
        if result:
            found = True
            assert "?:" not in result.rewrite
            break
    assert found, "Elvis operator should be supported"
