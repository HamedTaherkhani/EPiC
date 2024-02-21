from datasets import load_dataset
import re
from evaluate import load

class HumanEvalLoader:
    def __init__(self):
        self.human_eval = load_dataset("openai_humaneval")

    def get_human_eval(self):
        return self.human_eval

    def get_final_test_cases(self):
        final_test_cases = []
        for a_test in self.human_eval['test']:
            method_name = re.findall('def .*\(', a_test['prompt'])[0].replace('def ', '').replace('(', '')
            test = a_test['test'] + '\ncheck(' + method_name + ')'
            final_test_cases.append(test)
        return final_test_cases

    def load_cod_eval(self):
        code_eval_metric = load("code_eval")
        return code_eval_metric