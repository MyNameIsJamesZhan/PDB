from swesmith.bug_gen.procedural.php.base import php_parse


def test_php_parse_plain_code():
    """Plain PHP code should parse without needing a <?php tag."""
    src = "function foo() { return 1; }"
    tree, offset = php_parse(src)
    assert offset == 0
    assert tree.root_node is not None
    assert not tree.root_node.has_error


def test_php_parse_produces_function_node():
    """Parser should produce a function_definition node."""
    src = "function foo() { return 1; }"
    tree, _ = php_parse(src)
    children = [c.type for c in tree.root_node.children]
    assert "function_definition" in children
