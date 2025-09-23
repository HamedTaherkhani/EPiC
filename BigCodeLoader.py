import os

from datasets import load_dataset
import re
import json
import pickle
import numpy as np
def find_import_statements(python_code: str) -> list:
    # Regular expressions to match both 'import' and 'from ... import ...' statements
    import_pattern = r'^\s*import\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s+as\s+[a-zA-Z_][a-zA-Z0-9_]*)?'
    from_import_pattern = r'^\s*from\s+[a-zA-Z_][a-zA-Z0-9_]*\s+import\s+[a-zA-Z_][a-zA-Z0-9_]*'

    # Find all matches using the regex patterns
    imports = re.findall(import_pattern, python_code, re.MULTILINE)
    from_imports = re.findall(from_import_pattern, python_code, re.MULTILINE)

    # Combine both types of imports into a single list
    return imports + from_imports + ['import matplotlib as plt', 'import numpy as np', 'import pandas as pd']

def extract_function_signature(code_string):
    # Regular expression to capture the function signature
    signature_regex = r"def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)"

    # Search for the function signature in the given code string
    match = re.search(signature_regex, code_string)

    if match:
        return match.group(0)
    else:
        return None
def remove_specific_line(text):
    lines = text.split("\n")  # Split the string into a list of lines
    filtered_lines = [line for line in lines if line.strip() not in ('from task_module import task_func', 'from __main__ import task_func', 'from your_module import task_func', 'from task_module import task_func')]  # Remove the specific line
    return "\n".join(filtered_lines)  # Join the filtered lines back into a string

class BigCodeLoader:
    def __init__(self,hard=1):
        # ds = load_dataset("bigcode/bigcodebench", split="v0.1.2")
        ids = []
        if hard == 1:
            ds = load_dataset("bigcode/bigcodebench-hard", split="v0.1.2")
        else:
            data = []
            with open(f'{os.getcwd()}/loaders/bigcodebench_subset.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    data.append(json.loads(line.strip()))
            ids = [item['name'] for item in data]
            all_ds = load_dataset("bigcode/bigcodebench", split="v0.1.2")
            ds = []
            # ids_ = []
            for item in all_ds:
                if item['task_id'] in ids:
                    ds.append(item)
                    # ids_.append(item['task_id'])
            # self.ids = ids_
        self.prompts = []
        self.dataset = ds
        self.solutions = []
        # self.libs = []
        self.all_imports = []
        self.tests = []
        self.ids = []
        imports2 = ['import pandas as pd','import matplotlib.pyplot as plt','from sklearn.decomposition import PCA', 'import os', 'import shutil', 'import ftplib']
        for item in ds:
            # print(type(item['libs']))
            # print(item['libs'])
            # imports1 = [f'import {a}' for a in item['libs']]
            self.ids.append(item['task_id'])
            # print(imports1)
            # print(type(imports1))
            # print(extract_function_signature(item['complete_prompt']))
            imports = find_import_statements(item['complete_prompt']) + imports2
            # self.libs.append(item['libs'])
            self.prompts.append(item['complete_prompt'])
            self.tests.append(item['test'])
            for imp in imports:
                if imp not in self.all_imports:
                    self.all_imports.append(imp)
            # self.all_imports.extend(imports)
            # print(imports)
            # print('-----------------------------')
            if len(imports) == 0:
                print(item['complete_prompt'])
            # self.solutions.append('\n'.join(imports) + '\n' + extract_function_signature(item['complete_prompt'])+ ":\n" + item['canonical_solution'])
            imports_text = '\n'.join(imports) + '\n'
            try:
                sol = imports_text + item['instruct_prompt'].split('```')[1] + item['canonical_solution']
            except Exception as e:
                sol = imports_text+ item['instruct_prompt'] + item['canonical_solution']
            self.solutions.append(sol)
    def get_prompts(self):
        return self.prompts

    def get_tests(self):
        return self.tests
    #
    # def get_func_names(self):
    #     return self.func_names

    def get_dataset(self):
        return self.dataset

    def get_solutions(self):
        return self.solutions

    def get_ids(self):
        return self.ids

    def get_generated_tests_o3(self):
        with open('testcases/BigCodeBenchHard_o3-mini.pkl', 'rb') as fp:
            itemlist = pickle.load(fp)
        applyall = np.vectorize(remove_specific_line)
        items = [applyall(item[:4]) if len(item) != 0 else [] for index, item in enumerate(itemlist) ]
        return items

    def get_generated_tests_deepseek(self):
        with open('testcases/BigCodeBenchHard_deepseek.pkl', 'rb') as fp:
            itemlist = pickle.load(fp)
        items = [item[:4] if len(item) != 0 else [] for index, item in enumerate(itemlist)]
        return items

    def get_generated_tests_sonnet(self):
        with open('testcases/BigCodeBenchHard_claude-3-7-sonnet-20250219.pkl', 'rb') as fp:
            itemlist = pickle.load(fp)
        # applyall = np.vectorize(remove_specific_line)
        items = [item[:4] if len(item) != 0 else [] for index, item in enumerate(itemlist)]
        return items