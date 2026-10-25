"""Tests for multi-language AST parser."""

import pytest
from backend.ingestion.ast_parser import ASTParserManager
from backend.models.schemas import CodeUnitType


@pytest.fixture
def parser():
    return ASTParserManager()


def test_python_function_and_class_extraction(parser: ASTParserManager):
    code = '''
class Greeter:
    """Class docstring."""
    def __init__(self, prefix: str):
        self.prefix = prefix

    def greet(self, name: str) -> str:
        """Method docstring."""
        return f"{self.prefix} {name}"

@decorator_one
@decorator_two(option=True)
def calculate_sum(a: int, b: int = 10) -> int:
    """Calculate sum docstring."""
    return a + b

async def async_worker(task_id):
    pass
'''
    units = parser.extract_units(
        source_code=code,
        language="python",
        file_path="sample.py",
        repository="test/repo",
        commit_hash="abc1234",
    )

    names = {u.name: u for u in units}

    assert "Greeter" in names
    assert names["Greeter"].unit_type == CodeUnitType.CLASS
    assert names["Greeter"].docstring == "Class docstring."

    assert "Greeter.__init__" in names
    assert names["Greeter.__init__"].unit_type == CodeUnitType.METHOD
    assert names["Greeter.__init__"].parameters == ["self", "prefix"]

    assert "Greeter.greet" in names
    assert names["Greeter.greet"].unit_type == CodeUnitType.METHOD
    assert names["Greeter.greet"].docstring == "Method docstring."
    assert names["Greeter.greet"].parameters == ["self", "name"]

    assert "calculate_sum" in names
    assert names["calculate_sum"].unit_type == CodeUnitType.FUNCTION
    assert names["calculate_sum"].docstring == "Calculate sum docstring."
    assert "@decorator_one" in names["calculate_sum"].source_code

    assert "async_worker" in names
    assert names["async_worker"].unit_type == CodeUnitType.ASYNC_FUNCTION
    assert names["async_worker"].parameters == ["task_id"]


def test_javascript_typescript_extraction(parser: ASTParserManager):
    js_code = '''
class Service {
    execute(cmd) {
        return cmd;
    }
}

function processItems(items) {
    return items.length;
}

const compute = (x, y) => x + y;
'''
    units = parser.extract_units(
        source_code=js_code,
        language="javascript",
        file_path="service.js",
        repository="test/repo",
        commit_hash="abc1234",
    )

    names = {u.name: u for u in units}
    assert "Service" in names
    assert "Service.execute" in names
    assert "processItems" in names
    assert "compute" in names
    assert names["compute"].unit_type == CodeUnitType.FUNCTION


def test_go_extraction(parser: ASTParserManager):
    go_code = '''
package main

type Config struct {
    Port int
}

func (c *Config) GetPort() int {
    return c.Port
}

func Run() error {
    return nil
}
'''
    units = parser.extract_units(
        source_code=go_code,
        language="go",
        file_path="main.go",
        repository="test/repo",
        commit_hash="abc1234",
    )

    names = {u.name: u for u in units}
    assert "Config" in names
    assert "Config.GetPort" in names
    assert "Run" in names


def test_rust_extraction(parser: ASTParserManager):
    rs_code = '''
struct App;

impl App {
    fn start(&self) {
    }
}

fn helper() {
}
'''
    units = parser.extract_units(
        source_code=rs_code,
        language="rust",
        file_path="lib.rs",
        repository="test/repo",
        commit_hash="abc1234",
    )

    names = {u.name: u for u in units}
    assert "App" in names
    assert "App.start" in names
    assert "helper" in names


def test_java_extraction(parser: ASTParserManager):
    java_code = '''
public class Server {
    public Server() {}
    public void start() {}
}
'''
    units = parser.extract_units(
        source_code=java_code,
        language="java",
        file_path="Server.java",
        repository="test/repo",
        commit_hash="abc1234",
    )

    names = {u.name: u for u in units}
    assert "Server" in names
    assert "Server.Server" in names
    assert "Server.start" in names


def test_cpp_extraction(parser: ASTParserManager):
    cpp_code = '''
class Engine {
    void run() {}
};

int calculate(int x) {
    return x * 2;
}
'''
    units = parser.extract_units(
        source_code=cpp_code,
        language="cpp",
        file_path="engine.cpp",
        repository="test/repo",
        commit_hash="abc1234",
    )

    names = {u.name: u for u in units}
    assert "Engine" in names
    assert "calculate" in names
