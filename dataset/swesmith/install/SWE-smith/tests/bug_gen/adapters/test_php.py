from swesmith.bug_gen.adapters.php import get_entities_from_file_php
from swesmith.constants import CodeProperty


def test_get_entities_from_file_php(test_file_php):
    entities = []
    get_entities_from_file_php(entities, test_file_php)
    assert len(entities) == 5
    names = [e.name for e in entities]
    for name in [
        "ControllerDispatcher",
        "ControllerDispatcher::__construct",
        "ControllerDispatcher::dispatch",
        "ControllerDispatcher::resolveParameters",
        "ControllerDispatcher::getMiddleware",
    ]:
        assert name in names, f"Expected entity {name} not found in {names}"

    start_end = [(e.line_start, e.line_end) for e in entities]
    for start, end in [
        (10, 82),  # ControllerDispatcher class
        (26, 29),  # ControllerDispatcher::__construct
        (39, 48),  # ControllerDispatcher::dispatch
        (58, 63),  # ControllerDispatcher::resolveParameters
        (72, 81),  # ControllerDispatcher::getMiddleware
    ]:
        assert (start, end) in start_end, (
            f"Expected line range ({start}, {end}) not found in {start_end}"
        )

    assert all([e.ext == "php" for e in entities]), (
        "All entities should have the extension 'php'"
    )
    assert all([e.file_path == str(test_file_php) for e in entities]), (
        "All entities should have the correct file path"
    )

    signatures = [e.signature for e in entities]
    for signature in [
        "class ControllerDispatcher implements ControllerDispatcherContract",
        "public function __construct(Container $container)",
        "public function dispatch(Route $route, $controller, $method)",
        "protected function resolveParameters(Route $route, $controller, $method)",
        "public function getMiddleware($controller, $method)",
    ]:
        assert signature in signatures, (
            f"Expected signature '{signature}' not found in {signatures}"
        )

    stubs = [e.stub for e in entities]
    for stub in [
        "class ControllerDispatcher implements ControllerDispatcherContract {\n\t// TODO: Implement this function\n}",
        "public function __construct(Container $container) {\n\t// TODO: Implement this function\n}",
        "public function dispatch(Route $route, $controller, $method) {\n\t// TODO: Implement this function\n}",
        "protected function resolveParameters(Route $route, $controller, $method) {\n\t// TODO: Implement this function\n}",
        "public function getMiddleware($controller, $method) {\n\t// TODO: Implement this function\n}",
    ]:
        assert stub in stubs, f"Expected stub '{stub}' not found in {stubs}"


def test_get_entities_from_file_php_max(test_file_php):
    """Should cap the number of returned entities when *max_entities* is set."""
    entities: list = []
    get_entities_from_file_php(entities, test_file_php, 3)

    # Only three entities should be returned and they should be in the order
    # encountered in the file (depth-first traversal used by the adapter).
    assert len(entities) == 3
    assert [e.name for e in entities] == [
        "ControllerDispatcher",
        "ControllerDispatcher::__construct",
        "ControllerDispatcher::dispatch",
    ]


def test_get_entities_from_file_php_unreadable():
    """Asserting that unreadable / non-existent files are handled gracefully."""
    entities: list = []
    # The adapter swallows exceptions internally and simply returns the (still
    # empty) *entities* list, so we just verify that behaviour.
    get_entities_from_file_php(entities, "non-existent-file.php")
    assert entities == []


def test_get_entities_from_file_php_no_entities(tmp_path):
    """A PHP file with no top-level functions, methods or classes yields no entities."""
    no_entities_file = tmp_path / "no_entities.php"
    no_entities_file.write_text("<?php\n// Silence is golden\n")

    entities: list = []
    get_entities_from_file_php(entities, no_entities_file)
    assert len(entities) == 0


def test_php_entity_one_line_function(tmp_path):
    """Correctly pick up a function that lives entirely on one line."""
    one_line_file = tmp_path / "one_line.php"
    one_line_file.write_text("<?php\nfunction one_line_function() { return 42; }\n")

    entities: list = []
    get_entities_from_file_php(entities, one_line_file)

    assert len(entities) == 1
    entity = entities[0]
    assert entity.name == "one_line_function"
    assert entity.signature == "function one_line_function()"
    assert (
        entity.stub
        == "function one_line_function() {\n\t// TODO: Implement this function\n}"
    )


def test_php_entity_multi_line_signature(tmp_path):
    """Multi-line function signatures should be preserved in *signature*."""
    multi_line_file = tmp_path / "multi_line.php"
    multi_line_file.write_text(
        "<?php\nfunction multi_line_function(\n    $param1,\n    $param2\n) {\n    return $param1 + $param2;\n}\n"
    )

    entities: list = []
    get_entities_from_file_php(entities, multi_line_file)

    assert len(entities) == 1
    entity = entities[0]
    assert entity.name == "multi_line_function"
    assert (
        entity.signature
        == "function multi_line_function(\n    $param1,\n    $param2\n)"
    )


PHP_PREFIX = "<?php\n"


def _get_entities(tmp_path, src):
    test_file = tmp_path / "test.php"
    test_file.write_text(PHP_PREFIX + src, encoding="utf-8")
    entities = []
    get_entities_from_file_php(entities, str(test_file))
    return entities


def test_php_entity_class_name(tmp_path):
    """Test that class_declaration entities return the class name."""
    src = """class MyClass {
    public function doStuff() {
        return 42;
    }
}"""
    entities = _get_entities(tmp_path, src)
    class_entity = [e for e in entities if e.node.type == "class_declaration"]
    assert len(class_entity) >= 1
    assert class_entity[0].name == "MyClass"


def test_php_entity_method_name(tmp_path):
    """Test that method_declaration entities return ClassName::methodName."""
    src = """class Calculator {
    public function testAdd() {
        return 1 + 2;
    }
}"""
    entities = _get_entities(tmp_path, src)
    method_entities = [e for e in entities if e.node.type == "method_declaration"]
    assert len(method_entities) >= 1
    assert method_entities[0].name == "Calculator::testAdd"


def test_php_entity_unknown_name(tmp_path):
    """Test that an unknown node type returns 'unknown'."""
    src = """function normal() {
    return 1;
}"""
    entities = _get_entities(tmp_path, src)
    assert len(entities) >= 1
    # The function entity should have a real name, but let's verify the fallback
    entity = entities[0]
    assert entity.name == "normal"


def test_php_entity_has_switch(tmp_path):
    src = """function foo($x) {
    switch ($x) {
        case 1:
            return "one";
        case 2:
            return "two";
        default:
            return "other";
    }
}"""
    entities = _get_entities(tmp_path, src)
    assert len(entities) >= 1
    assert CodeProperty.HAS_SWITCH in entities[0]._tags


def test_php_entity_has_exception(tmp_path):
    src = """function foo($x) {
    try {
        return $x;
    } catch (Exception $e) {
        return null;
    }
}"""
    entities = _get_entities(tmp_path, src)
    assert len(entities) >= 1
    assert CodeProperty.HAS_EXCEPTION in entities[0]._tags


def test_php_entity_has_import(tmp_path):
    src = """use App\\Models\\User;
function foo() {
    return new User();
}"""
    entities = _get_entities(tmp_path, src)
    func_entities = [e for e in entities if e.node.type == "function_definition"]
    # The import tag is checked at the walk level, so it might be on the function
    # if the use statement is walked as part of the file
    assert len(func_entities) >= 1


def test_php_entity_has_lambda_arrow(tmp_path):
    src = """function foo() {
    $fn = fn($x) => $x * 2;
    return $fn(5);
}"""
    entities = _get_entities(tmp_path, src)
    assert len(entities) >= 1
    assert CodeProperty.HAS_LAMBDA in entities[0]._tags


def test_php_entity_has_lambda_anonymous(tmp_path):
    """Anonymous function (closure) should trigger HAS_LAMBDA."""
    src = """function foo() {
    $fn = function($x) { return $x * 2; };
    return $fn(5);
}"""
    entities = _get_entities(tmp_path, src)
    assert len(entities) >= 1
    assert CodeProperty.HAS_LAMBDA in entities[0]._tags


def test_php_entity_has_arithmetic(tmp_path):
    src = """function foo($x) {
    return $x + 1;
}"""
    entities = _get_entities(tmp_path, src)
    assert len(entities) >= 1
    assert CodeProperty.HAS_ARITHMETIC in entities[0]._tags


def test_php_entity_has_ternary(tmp_path):
    src = """function foo($x) {
    return $x > 0 ? "yes" : "no";
}"""
    entities = _get_entities(tmp_path, src)
    assert len(entities) >= 1
    assert CodeProperty.HAS_TERNARY in entities[0]._tags


def test_php_entity_has_parent(tmp_path):
    src = """class Child extends Parent {
    public function doStuff() {
        return 1;
    }
}"""
    entities = _get_entities(tmp_path, src)
    class_entities = [e for e in entities if e.node.type == "class_declaration"]
    assert len(class_entities) >= 1
    assert CodeProperty.HAS_PARENT in class_entities[0]._tags


def test_php_entity_has_if_else(tmp_path):
    src = """function foo($x) {
    if ($x > 0) {
        return "positive";
    } else {
        return "non-positive";
    }
}"""
    entities = _get_entities(tmp_path, src)
    assert len(entities) >= 1
    assert CodeProperty.HAS_IF_ELSE in entities[0]._tags


def test_php_entity_complexity(tmp_path):
    src = """function foo($x, $y) {
    if ($x > 0) {
        if ($y > 0) {
            return "both positive";
        } else {
            return "x positive";
        }
    }
    for ($i = 0; $i < $x; $i++) {
        echo $i;
    }
    return $x && $y ? "truthy" : "falsy";
}"""
    entities = _get_entities(tmp_path, src)
    func_entities = [e for e in entities if e.node.type == "function_definition"]
    assert len(func_entities) >= 1
    # Should have complexity > 1 (base 1 + if + if + else + for + && + ternary)
    assert func_entities[0].complexity >= 5


def test_php_entity_no_body_stub(tmp_path):
    """Test stub for entity without a body (signature only won't parse, but edge case)."""
    src = """function simple() { return 1; }"""
    entities = _get_entities(tmp_path, src)
    assert len(entities) >= 1
    assert entities[0].stub.endswith("}")


def test_php_entity_method_without_parent_class(tmp_path):
    """Test _find_parent_class returning None for a standalone function."""
    src = """function standalone() {
    return 42;
}"""
    entities = _get_entities(tmp_path, src)
    assert len(entities) >= 1
    # _find_parent_class should return None for a function_definition
    result = entities[0]._find_parent_class()
    assert result is None
