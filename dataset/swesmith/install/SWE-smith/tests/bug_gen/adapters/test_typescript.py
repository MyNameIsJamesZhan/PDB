import warnings

import pytest

from swesmith.bug_gen.adapters.typescript import get_entities_from_file_ts


@pytest.fixture
def entities(test_file_ts):
    entities = []
    get_entities_from_file_ts(entities, test_file_ts)
    return entities


def test_get_entities_from_file_ts_count(entities):
    # Calculator class, constructor, add, multiply, greet, incrementCounter
    assert len(entities) >= 5


def test_get_entities_from_file_ts_max(test_file_ts):
    entities = []
    get_entities_from_file_ts(entities, test_file_ts, 3)
    assert len(entities) == 3


def test_get_entities_from_file_ts_names(entities):
    names = [e.name for e in entities]
    assert "Calculator" in names
    assert "greet" in names
    assert "add" in names


def test_get_entities_from_file_ts_extensions(entities):
    assert all(e.ext == "ts" for e in entities)


def test_get_entities_from_file_ts_file_paths(entities, test_file_ts):
    assert all(e.file_path == str(test_file_ts) for e in entities)


def test_get_entities_from_file_ts_no_functions(tmp_path):
    no_functions_file = tmp_path / "no_functions.ts"
    no_functions_file.write_text("// no functions\nconst x: number = 5;")
    entities = []
    get_entities_from_file_ts(entities, no_functions_file)
    assert len(entities) == 0


def test_get_entities_from_file_ts_malformed(tmp_path):
    malformed_file = tmp_path / "malformed.ts"
    malformed_file.write_text("(malformed")
    entities = []
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        get_entities_from_file_ts(entities, malformed_file)
        assert any("Error encountered parsing" in str(w.message) for w in ws)


def test_get_entities_from_file_ts_with_type_annotations(tmp_path):
    ts_file = tmp_path / "typed.ts"
    ts_file.write_text("function add(a: number, b: number): number { return a + b; }")
    entities = []
    get_entities_from_file_ts(entities, ts_file)
    assert len(entities) == 1
    assert entities[0].name == "add"
    assert "number" in entities[0].signature


def test_get_entities_from_file_ts_class(tmp_path):
    ts_file = tmp_path / "class.ts"
    ts_file.write_text(
        """
class MyClass {
    myMethod(x: string): string {
        return x;
    }
}
    """.strip()
    )
    entities = []
    get_entities_from_file_ts(entities, ts_file)
    assert len(entities) == 2
    names = [e.name for e in entities]
    assert "MyClass" in names
    assert "myMethod" in names


def test_get_entities_from_file_ts_function_expression(tmp_path):
    ts_file = tmp_path / "func_expr.ts"
    ts_file.write_text("var myFunc = function(x: number): number { return x * 2; };")
    entities = []
    get_entities_from_file_ts(entities, ts_file)
    assert len(entities) == 1
    assert entities[0].name == "myFunc"


def test_get_entities_from_file_ts_complexity(tmp_path):
    ts_file = tmp_path / "complex.ts"
    ts_file.write_text(
        """
function complex(x: number): number {
    if (x > 0) {
        for (let i = 0; i < x; i++) {
            console.log(i);
        }
    } else {
        while (x < 0) {
            x++;
        }
    }
    return x;
}
    """.strip()
    )
    entities = []
    get_entities_from_file_ts(entities, ts_file)
    assert len(entities) == 1
    # base(1) + if(1) + else(1) + for(1) + while(1) = 5
    assert entities[0].complexity >= 4


def test_get_entities_from_file_ts_boolean_operators(tmp_path):
    ts_file = tmp_path / "bool.ts"
    ts_file.write_text(
        "function f(a: boolean, b: boolean): boolean { return a && b || !a; }"
    )
    entities = []
    get_entities_from_file_ts(entities, ts_file)
    assert len(entities) == 1
    assert entities[0].has_bool_op


def test_get_entities_from_file_ts_interface_ignored(tmp_path):
    """Interfaces should not be collected as entities."""
    ts_file = tmp_path / "interface.ts"
    ts_file.write_text(
        """
interface User {
    name: string;
    age: number;
}

function greet(user: User): string {
    return user.name;
}
    """.strip()
    )
    entities = []
    get_entities_from_file_ts(entities, ts_file)
    # Only the function should be collected, not the interface
    assert len(entities) == 1
    assert entities[0].name == "greet"


def test_get_entities_from_file_ts_abstract_methods_ignored(tmp_path):
    """Abstract methods should not be collected as entities."""
    ts_file = tmp_path / "abstract.ts"
    ts_file.write_text(
        """
abstract class AbstractClass {
    abstract abstractMethod(): void;
    abstract anotherAbstract(): string;
    
    concreteMethod(): void {
        return;
    }
}

class ConcreteClass {
    regularMethod(): void {
        return;
    }
}
    """.strip()
    )
    entities = []
    get_entities_from_file_ts(entities, ts_file)
    # Only concrete methods and concrete classes should be collected
    names = [e.name for e in entities]
    assert "abstractMethod" not in names
    assert "anotherAbstract" not in names
    assert "AbstractClass" not in names
    assert "concreteMethod" in names
    assert "ConcreteClass" in names
    assert "regularMethod" in names


def test_get_entities_from_file_ts_multiline_signature(tmp_path):
    """Multi-line function signatures should be correctly extracted without body."""
    ts_file = tmp_path / "multiline.ts"
    ts_file.write_text(
        """
class MyClass {
  public request({
    op: { id, type, path, input, signal },
    transformer,
    lastEventId,
  }: {
    op: Pick<Operation, 'id' | 'type' | 'path' | 'input' | 'signal'>;
    transformer: CombinedDataTransformer;
    lastEventId?: string;
  }) {
    return 'test';
  }
}
    """.strip()
    )
    entities = []
    get_entities_from_file_ts(entities, ts_file)
    method = next((e for e in entities if e.name == "request"), None)
    assert method is not None
    signature = method.signature
    # Signature should include the full parameter list but NOT the body
    assert "op: { id, type, path, input, signal }" in signature
    assert "transformer" in signature
    assert "lastEventId" in signature
    # Check for complex type annotations
    assert "Pick<Operation" in signature
    assert "CombinedDataTransformer" in signature
    assert "lastEventId?: string" in signature
    # Should NOT include body content
    assert "return 'test'" not in signature
    assert "return" not in signature
    # Should end with closing paren, not opening brace
    assert signature.rstrip().endswith(")")


def test_get_entities_from_file_ts_generator_functions(tmp_path):
    """Test that generator functions (function*) are collected."""
    ts_file = tmp_path / "generator.ts"
    ts_file.write_text(
        """
function* myGenerator() {
  yield 1;
  yield 2;
}

class MyClass {
  *generatorMethod() {
    yield 'test';
  }
  
  regularMethod() {
    return 'test';
  }
}
    """.strip()
    )
    entities = []
    get_entities_from_file_ts(entities, ts_file)
    names = [e.name for e in entities]
    assert "myGenerator" in names
    assert "generatorMethod" in names
    assert "regularMethod" in names
    # Verify generator function is tagged as function
    generator = next(e for e in entities if e.name == "myGenerator")
    assert generator.is_function


def test_get_entities_from_file_ts_try_catch(tmp_path):
    """Test try/catch/throw detection."""
    ts_file = tmp_path / "trycatch.ts"
    ts_file.write_text(
        """
function riskyOperation(): void {
    try {
        throw new Error("oops");
    } catch (e) {
        console.log(e);
    }
}
    """.strip()
    )
    entities = []
    get_entities_from_file_ts(entities, ts_file)
    assert len(entities) == 1
    assert entities[0].has_exception


def test_get_entities_from_file_ts_class_inheritance(tmp_path):
    """Test class with extends (HAS_PARENT)."""
    ts_file = tmp_path / "inheritance.ts"
    ts_file.write_text(
        """
class Animal {
    name: string;
}

class Dog extends Animal {
    bark(): void {
        console.log("woof");
    }
}
    """.strip()
    )
    entities = []
    get_entities_from_file_ts(entities, ts_file)
    names = [e.name for e in entities]
    assert "Dog" in names
    dog = next(e for e in entities if e.name == "Dog")
    assert dog.has_parent


def test_get_entities_from_file_ts_ternary(tmp_path):
    """Test ternary expression detection."""
    ts_file = tmp_path / "ternary.ts"
    ts_file.write_text("function abs(x: number): number { return x >= 0 ? x : -x; }")
    entities = []
    get_entities_from_file_ts(entities, ts_file)
    assert len(entities) == 1
    assert entities[0].has_ternary


def test_get_entities_from_file_ts_stub(tmp_path):
    """Test stub generation."""
    ts_file = tmp_path / "stub.ts"
    ts_file.write_text("function hello(): string { return 'hello'; }")
    entities = []
    get_entities_from_file_ts(entities, ts_file)
    assert len(entities) == 1
    stub = entities[0].stub
    assert "function hello(): string" in stub
    assert "TODO" in stub


def test_get_entities_from_file_ts_for_in_of(tmp_path):
    """Test for-in and for-of loop detection."""
    ts_file = tmp_path / "forin.ts"
    ts_file.write_text(
        """
function iterate(arr: number[]): void {
    for (const item of arr) {
        console.log(item);
    }
    for (const key in arr) {
        console.log(key);
    }
}
    """.strip()
    )
    entities = []
    get_entities_from_file_ts(entities, ts_file)
    assert len(entities) == 1
    assert entities[0].has_loop


def test_get_entities_from_file_ts_do_while(tmp_path):
    """Test do-while loop detection."""
    ts_file = tmp_path / "dowhile.ts"
    ts_file.write_text(
        """
function countdown(n: number): void {
    do {
        console.log(n);
        n--;
    } while (n > 0);
}
    """.strip()
    )
    entities = []
    get_entities_from_file_ts(entities, ts_file)
    assert len(entities) == 1
    assert entities[0].has_loop


def test_get_entities_from_file_ts_switch(tmp_path):
    """Test switch statement complexity."""
    ts_file = tmp_path / "switch.ts"
    ts_file.write_text(
        """
function grade(score: number): string {
    switch (true) {
        case score >= 90: return 'A';
        case score >= 80: return 'B';
        default: return 'C';
    }
}
    """.strip()
    )
    entities = []
    get_entities_from_file_ts(entities, ts_file)
    assert len(entities) == 1
    # base(1) + switch(1) = 2
    assert entities[0].complexity >= 2
