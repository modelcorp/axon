#!/usr/bin/env python3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add recipes folder to sys.path since it's outside the axon package
_recipes_path = Path(__file__).parent.parent.parent.parent / "recipes"
if str(_recipes_path) not in sys.path:
    sys.path.insert(0, str(_recipes_path))

from tools.code_tools import PythonInterpreter  # noqa: E402
from tools.web_tools import GoogleSearchTool  # noqa: E402

from axon.tools.types import ToolOutput  # noqa: E402

# Mock response data for Google Search API
MOCK_GOOGLE_SEARCH_RESPONSE = {
    "items": [
        {
            "title": "Python (programming language) - Wikipedia",
            "snippet": "Python is a high-level, general-purpose programming language.",
            "link": "https://en.wikipedia.org/wiki/Python_(programming_language)",
        },
        {
            "title": "Welcome to Python.org",
            "snippet": "The official home of the Python Programming Language.",
            "link": "https://www.python.org/",
        },
        {
            "title": "Python Tutorial - W3Schools",
            "snippet": "Python is a popular programming language. Python can be used on a server to create web applications.",
            "link": "https://www.w3schools.com/python/",
        },
    ]
}

# Test code for Python interpreter
python_test_cases = [
    {
        "name": "Basic Arithmetic",
        "code": """
print("Testing basic arithmetic...")
x = 10
y = 5
print(f"Addition: {x + y}")
print(f"Subtraction: {x - y}")
print(f"Multiplication: {x * y}")
print(f"Division: {x / y}")
print(f"Integer Division: {x // y}")
print(f"Modulo: {x % y}")
print(f"Power: {x ** y}")
""",
        "expected_stdout": "Testing basic arithmetic...\nAddition: 15\nSubtraction: 5\nMultiplication: 50\nDivision: 2.0\nInteger Division: 2\nModulo: 0\nPower: 100000",
    },
    {
        "name": "List Operations",
        "code": """
print("Testing list operations...")
numbers = [1, 2, 3, 4, 5]
print(f"Original list: {numbers}")
print(f"Sum: {sum(numbers)}")
print(f"Average: {sum(numbers)/len(numbers)}")
print(f"Max: {max(numbers)}")
print(f"Min: {min(numbers)}")
squared = [x**2 for x in numbers]
print(f"Squared: {squared}")
""",
        "expected_stdout": "Testing list operations...\nOriginal list: [1, 2, 3, 4, 5]\nSum: 15\nAverage: 3.0\nMax: 5\nMin: 1\nSquared: [1, 4, 9, 16, 25]",
    },
    {
        "name": "Error Handling",
        "code": """
print("Testing error handling...")
try:
    result = 1/0
except ZeroDivisionError as e:
    print(f"Caught error: {e}")
try:
    undefined_var
except NameError as e:
    print(f"Caught error: {e}")
""",
        "expected_stdout": "Testing error handling...\nCaught error: division by zero\nCaught error: name 'undefined_var' is not defined",
    },
    {
        "name": "File Operations",
        "code": """
print("Testing file operations...")
import os, tempfile
test_file = os.path.join(tempfile.gettempdir(), "axon_test_output.txt")
with open(test_file, "w") as f:
    f.write("Hello, World!")
with open(test_file, "r") as f:
    content = f.read()
os.remove(test_file)
print(f"File content: {content}")
print("File operations completed successfully")
""",
        "expected_stdout": "Testing file operations...\nFile content: Hello, World!\nFile operations completed successfully",
    },
]

# Test code for async Python interpreter
python_async_test_cases = [
    {
        "name": "Basic Async",
        "code": """
import asyncio
import time

async def count():
    for i in range(3):
        print(f"Count: {i}")
        await asyncio.sleep(0.1)

print("Starting async test...")
asyncio.run(count())
print("Async test complete!")
""",
        "expected_stdout": "Starting async test...\nCount: 0\nCount: 1\nCount: 2\nAsync test complete!",
    },
    {
        "name": "Multiple Async Tasks",
        "code": """
import asyncio
import time

async def task(name, delay):
    print(f"Task {name} started")
    await asyncio.sleep(delay)
    print(f"Task {name} completed")

async def main():
    print("Starting multiple tasks...")
    tasks = [
        task("A", 0.1),
        task("B", 0.2),
        task("C", 0.3)
    ]
    await asyncio.gather(*tasks)
    print("All tasks completed!")

asyncio.run(main())
""",
        "expected_stdout": "Starting multiple tasks...\nTask A started\nTask B started\nTask C started\nTask A completed\nTask B completed\nTask C completed\nAll tasks completed!",
    },
]

# Test queries for search tools
search_test_cases = {
    "google_search": [
        {
            "name": "Basic Search",
            "query": "What is Python programming?",
            "expected_fields": ["title", "snippet", "link"],
        },
        {
            "name": "Technical Search",
            "query": "Python async await syntax example",
            "expected_fields": ["title", "snippet", "link"],
        },
    ],
    # "tavily_search": [{"name": "News Search", "query": "Latest developments in AI", "expected_fields": ["title", "snippet", "url"]}, {"name": "Technical Search", "query": "Python type hints tutorial", "expected_fields": ["title", "snippet", "url"]}],
    # "tavily_extract": [{"name": "Python.org", "url": "https://www.python.org/about/", "expected_fields": ["title", "text"]}, {"name": "Python Docs", "url": "https://docs.python.org/3/tutorial/", "expected_fields": ["title", "text"]}],
    # "firecrawl": [{"name": "Python.org", "url": "https://www.python.org", "expected_fields": ["title", "text", "links"]}, {"name": "Python Docs", "url": "https://docs.python.org/3/", "expected_fields": ["title", "text", "links"]}],
}


def validate_tool_output(result: ToolOutput, expected_fields: list | None = None) -> bool:
    """Validate the tool output has the expected structure and content."""
    if result.error:
        print(f"Error in tool execution: {result.error}")
        return False

    if expected_fields:
        if isinstance(result.output, dict):
            missing_fields = [field for field in expected_fields if field not in result.output]
            if missing_fields:
                print(f"Missing expected fields: {missing_fields}")
                return False
        else:
            print("Expected dictionary output with fields")
            return False

    return True


def test_python_tool():
    """Test the Python interpreter tool synchronously and asynchronously."""
    print("\nTesting Python interpreter tool...")

    # Create the tool directly
    python_tool = PythonInterpreter()

    # Test sync cases
    print("\nTesting sync execution:")
    for test_case in python_test_cases:
        print(f"\nRunning test: {test_case['name']}")
        print("Code:")
        print(test_case["code"])

        result = python_tool.forward(test_case["code"])
        print("Result:")
        print(f"Error: {result.error}")
        print(f"Output: {result.output}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")

        # Handle None values properly
        actual_stdout = result.stdout.strip() if result.stdout else ""
        expected_stdout = test_case["expected_stdout"].strip()

        if result.error:
            print(f"Test failed with error: {result.error}")
        elif actual_stdout == expected_stdout:
            print("Test passed!")
        else:
            print("Test failed! Output doesn't match expected")
            print("Expected:")
            print(expected_stdout)
            print("Actual:")
            print(actual_stdout)

    # Note: async execution tests removed — PythonInterpreter does not support async_forward


@pytest.mark.parametrize("tool_name", list(search_test_cases.keys()))
@patch.dict("os.environ", {"GOOGLE_SEARCH_SECRET_KEY": "fake_key", "GOOGLE_SEARCH_ENGINE_ID": "fake_engine_id"})
def test_search_tool(tool_name: str):
    """Test a search tool with multiple test cases."""
    print(f"\nTesting {tool_name} tool...")

    # Create mock response
    mock_response = MagicMock()
    mock_response.is_success = True
    mock_response.json.return_value = MOCK_GOOGLE_SEARCH_RESPONSE

    # Create the tool directly based on name
    tool_classes = {
        "google_search": GoogleSearchTool,
    }

    with patch("httpx.Client.get", return_value=mock_response):
        tool = tool_classes[tool_name]()

        # Run test cases
        for test_case in search_test_cases[tool_name]:
            print(f"\nRunning test: {test_case['name']}")
            print(f"Query/URL: {test_case.get('query', test_case.get('url'))}")

            # Execute tool
            if "query" in test_case:
                result = tool.forward(test_case["query"])
            else:
                result = tool.forward(test_case["url"])

            print("Result:")
            print(f"Error: {result.error}")
            print(f"Output: {result.output}")

            # Validate output - GoogleSearchTool returns {link: snippet} dict
            assert result.error is None, f"Test '{test_case['name']}' returned error: {result.error}"
            assert isinstance(result.output, dict), f"Test '{test_case['name']}' output is not a dict"
            assert len(result.output) > 0, f"Test '{test_case['name']}' returned empty results"
            # Verify the output contains links as keys and snippets as values
            for link, snippet in result.output.items():
                assert isinstance(link, str) and link.startswith("http"), f"Invalid link: {link}"
                assert isinstance(snippet, str) and len(snippet) > 0, f"Invalid snippet for {link}"
