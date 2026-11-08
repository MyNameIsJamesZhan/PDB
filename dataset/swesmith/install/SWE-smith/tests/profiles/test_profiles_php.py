import os
import pytest
import tempfile

from swesmith.profiles.php import (
    PhpProfile,
    Dbalacb68b38,
    Monolog6db20ca0,
    Guzzlefb92d95f,
    parse_log_phpunit_testdox,
)
from swesmith.constants import ENV_NAME
from swebench.harness.constants import TestStatus


def make_dummy_php_profile():
    class DummyPhpProfile(PhpProfile):
        owner = "dummy"
        repo = "dummyrepo"
        commit = "deadbeefcafebabe"

        @property
        def dockerfile(self):
            return "FROM php:8.3\nRUN echo hello"

        def log_parser(self, log):
            return parse_log_phpunit_testdox(log)

    return DummyPhpProfile()


def _write_file(base, relpath, content):
    full = os.path.join(base, relpath)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(content)


def _make_profile_with_clone(tmp_path):
    profile = make_dummy_php_profile()

    def fake_clone(dest=None):
        return str(tmp_path), False

    profile.clone = fake_clone
    return profile


def _make_profile_with_cache(cache):
    profile = make_dummy_php_profile()
    profile._fqcn_to_file_cache = cache
    return profile


def test_parse_log_phpunit_testdox_with_class_header():
    log = (
        "Connection (Doctrine\\DBAL\\Tests\\ConnectionTest)\n"
        " ✔ Prepare\n"
        " ✔ Query\n"
        " ✘ Exec"
    )
    result = parse_log_phpunit_testdox(log)
    assert (
        result["Doctrine\\DBAL\\Tests\\ConnectionTest::Prepare"]
        == TestStatus.PASSED.value
    )
    assert (
        result["Doctrine\\DBAL\\Tests\\ConnectionTest::Query"]
        == TestStatus.PASSED.value
    )
    assert (
        result["Doctrine\\DBAL\\Tests\\ConnectionTest::Exec"] == TestStatus.FAILED.value
    )


def test_parse_log_phpunit_testdox_multiple_classes():
    log = (
        "Connection (App\\Tests\\ATest)\n"
        " ✔ Execute\n"
        "\n"
        "Statement (App\\Tests\\BTest)\n"
        " ✔ Execute\n"
    )
    result = parse_log_phpunit_testdox(log)
    assert "App\\Tests\\ATest::Execute" in result
    assert "App\\Tests\\BTest::Execute" in result
    assert len(result) == 2


def test_parse_log_phpunit_testdox_skipped():
    log = "Foo (App\\Tests\\FooTest)\n ↩ Some skipped test"
    result = parse_log_phpunit_testdox(log)
    assert result["App\\Tests\\FooTest::Some skipped test"] == TestStatus.SKIPPED.value


def test_parse_log_phpunit_testdox_no_class_header():
    log = " ✔ Bare test"
    result = parse_log_phpunit_testdox(log)
    assert result["Bare test"] == TestStatus.PASSED.value


def test_parse_log_phpunit_testdox_empty():
    result = parse_log_phpunit_testdox("")
    assert result == {}


def test_build_map_basic_test_file(tmp_path):
    _write_file(
        tmp_path,
        "tests/ConnectionTest.php",
        "<?php\nnamespace App\\Tests;\nclass ConnectionTest {\n    public function testPrepare() {}\n}\n",
    )
    profile = _make_profile_with_clone(tmp_path)
    result = profile._build_fqcn_to_file_map()
    assert result["App\\Tests\\ConnectionTest"] == "tests/ConnectionTest.php"


def test_build_map_no_namespace(tmp_path):
    _write_file(
        tmp_path,
        "tests/FooTest.php",
        "<?php\nclass FooTest {\n    function testSomething() {}\n}\n",
    )
    profile = _make_profile_with_clone(tmp_path)
    result = profile._build_fqcn_to_file_map()
    assert result["FooTest"] == "tests/FooTest.php"


def test_build_map_vendor_skipped(tmp_path):
    _write_file(
        tmp_path,
        "vendor/pkg/tests/SomeTest.php",
        "<?php\nnamespace Vendor\\Tests;\nclass SomeTest {}\n",
    )
    profile = _make_profile_with_clone(tmp_path)
    result = profile._build_fqcn_to_file_map()
    assert result == {}


def test_build_map_non_test_file_ignored(tmp_path):
    _write_file(
        tmp_path,
        "src/Connection.php",
        "<?php\nnamespace App;\nclass Connection {}\n",
    )
    profile = _make_profile_with_clone(tmp_path)
    result = profile._build_fqcn_to_file_map()
    assert result == {}


def test_build_map_test_dir_convention(tmp_path):
    _write_file(
        tmp_path,
        "Test/Unit/Helper.php",
        "<?php\nnamespace App\\Test\\Unit;\nclass Helper {}\n",
    )
    profile = _make_profile_with_clone(tmp_path)
    result = profile._build_fqcn_to_file_map()
    assert result["App\\Test\\Unit\\Helper"] == "Test/Unit/Helper.php"


def test_build_map_same_method_different_classes(tmp_path):
    _write_file(
        tmp_path,
        "tests/ATest.php",
        "<?php\nnamespace App\\Tests;\nclass ATest {\n    public function testExecute() {}\n}\n",
    )
    _write_file(
        tmp_path,
        "tests/BTest.php",
        "<?php\nnamespace App\\Tests;\nclass BTest {\n    public function testExecute() {}\n}\n",
    )
    profile = _make_profile_with_clone(tmp_path)
    result = profile._build_fqcn_to_file_map()
    assert result["App\\Tests\\ATest"] == "tests/ATest.php"
    assert result["App\\Tests\\BTest"] == "tests/BTest.php"


def test_get_test_files_basic():
    cache = {
        "App\\Tests\\ConnectionTest": "tests/ConnectionTest.php",
        "App\\Tests\\FormatterTest": "tests/FormatterTest.php",
    }
    profile = _make_profile_with_cache(cache)
    instance = {
        "instance_id": "dummy__dummyrepo.deadbeef.1",
        "FAIL_TO_PASS": ["App\\Tests\\ConnectionTest::Prepare"],
        "PASS_TO_PASS": [
            "App\\Tests\\ConnectionTest::Query",
            "App\\Tests\\FormatterTest::Format output",
        ],
    }
    f2p, p2p = profile.get_test_files(instance)
    assert set(f2p) == {"tests/ConnectionTest.php"}
    assert set(p2p) == {"tests/ConnectionTest.php", "tests/FormatterTest.php"}


def test_get_test_files_missing_names():
    cache = {"App\\Tests\\ConnectionTest": "tests/ConnectionTest.php"}
    profile = _make_profile_with_cache(cache)
    instance = {
        "instance_id": "dummy__dummyrepo.deadbeef.1",
        "FAIL_TO_PASS": ["App\\Tests\\Missing::Not in cache"],
        "PASS_TO_PASS": ["App\\Tests\\AlsoMissing::Also missing"],
    }
    f2p, p2p = profile.get_test_files(instance)
    assert f2p == []
    assert p2p == []


def test_get_test_files_assertion_error():
    profile = _make_profile_with_cache({})
    with pytest.raises(AssertionError):
        profile.get_test_files({"instance_id": "dummy__dummyrepo.deadbeef.1"})


def test_get_test_files_cache_reuse():
    profile = make_dummy_php_profile()
    clone_count = 0

    def counting_clone(dest=None):
        nonlocal clone_count
        clone_count += 1
        d = tempfile.mkdtemp()
        return d, True

    profile.clone = counting_clone

    instance = {
        "instance_id": "dummy__dummyrepo.deadbeef.1",
        "FAIL_TO_PASS": ["test_a"],
        "PASS_TO_PASS": ["test_b"],
    }
    profile.get_test_files(instance)
    profile.get_test_files(instance)
    assert clone_count == 1


def test_dbal_dockerfile():
    profile = Dbalacb68b38()
    assert "php:8.3" in profile.dockerfile
    assert f"/{ENV_NAME}" in profile.dockerfile
    assert "composer" in profile.dockerfile


def test_monolog_dockerfile():
    profile = Monolog6db20ca0()
    assert "php:8.3" in profile.dockerfile
    assert "ext-mongodb" in profile.dockerfile


def test_guzzle_dockerfile():
    profile = Guzzlefb92d95f()
    assert "php:8.3" in profile.dockerfile
    assert "composer install" in profile.dockerfile


def test_php_profile_defaults():
    profile = make_dummy_php_profile()
    assert profile.org_dh == "swebench"
    assert profile.exts == [".php"]
    assert "phpunit" in profile.test_cmd


def test_build_map_unreadable_file(tmp_path):
    """Files that can't be read (OSError/UnicodeDecodeError) are skipped."""
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    bad_file = test_dir / "BadTest.php"
    bad_file.write_bytes(b"<?php\n\xff\xfe class BadTest {}\n")

    profile = _make_profile_with_clone(tmp_path)
    result = profile._build_fqcn_to_file_map()
    assert isinstance(result, dict)


def test_build_map_test_prefixed_file(tmp_path):
    """Files starting with 'test' should be picked up even outside tests/ dir."""
    _write_file(
        tmp_path,
        "src/testHelper.php",
        "<?php\nclass TestHelper {}\n",
    )
    profile = _make_profile_with_clone(tmp_path)
    result = profile._build_fqcn_to_file_map()
    assert result["TestHelper"] == "src/testHelper.php"


def test_get_test_files_partial_cache_match():
    """Some test names in cache, some not."""
    cache = {
        "App\\Tests\\ConnectionTest": "tests/ConnectionTest.php",
    }
    profile = _make_profile_with_cache(cache)
    instance = {
        "instance_id": "dummy__dummyrepo.deadbeef.1",
        "FAIL_TO_PASS": [
            "App\\Tests\\ConnectionTest::Prepare",
            "App\\Tests\\Missing::Not in cache",
        ],
        "PASS_TO_PASS": [
            "App\\Tests\\ConnectionTest::Query",
            "App\\Tests\\AlsoMissing::Also missing",
        ],
    }
    f2p, p2p = profile.get_test_files(instance)
    assert set(f2p) == {"tests/ConnectionTest.php"}
    assert set(p2p) == {"tests/ConnectionTest.php"}
