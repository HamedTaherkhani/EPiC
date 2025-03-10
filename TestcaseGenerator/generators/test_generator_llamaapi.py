from TestcaseGenerator.generators.py_generate import PY_TEST_GENERATION_FEW_SHOT, PY_TEST_GENERATION_COMPLETION_INSTRUCTION
import json
from llamaapi import LlamaAPI
import os
import ast
from typing import List
from dotenv import load_dotenv
import time
load_dotenv()
api_key = os.getenv('llama_api_key')


def parse_tests(tests: str) -> List[str]:
    tests = [test.strip() for test in tests.splitlines() if "assert" in test]
    # tests = [test.removeprefix("self.") for test in tests if test.startswith("self.")]
    return tests

def py_is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False

def generate_tests_llamaapi(model_name, func_sig):
    # prompt = f"{PY_TEST_GENERATION_COMPLETION_INSTRUCTION}\n\n{PY_TEST_GENERATION_FEW_SHOT}"
    llama = LlamaAPI(api_key)
    time.sleep(1)
    api_request_json = {
        "model": model_name,
        "max_tokens": 2000,
        "messages": [
            {"role": "system", "content": PY_TEST_GENERATION_COMPLETION_INSTRUCTION + ". Don't use python unittest library. Create separate assertions like given in the example."},
            {"role": "user", "content": f"{PY_TEST_GENERATION_FEW_SHOT}\n\n[func signature]:\n{func_sig}\n\n[think]:"},
        ],
        "stream": False,
    }
    try:
        response = llama.run(api_request_json)
    except Exception:
        return []

    response = response.json()
    content = response['choices'][0]['message']['content']
    # print(content)
    all_tests = parse_tests(content)  # type: ignore
    valid_tests = [test for test in all_tests if py_is_syntax_valid(test)]
    return valid_tests

