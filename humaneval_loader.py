from datasets import load_dataset
import re
from evaluate import load

class HumanEvalLoader:
    def __init__(self, instances=None):
        self.human_eval = load_dataset("openai_humaneval")
        if instances is not None:
            self.human_eval['test'] = [inst for idx,inst in enumerate(self.human_eval['test']) if idx in instances]

    def get_human_eval(self):
        return self.human_eval

    def get_final_test_cases(self):
        final_test_cases = []
        for a_test in self.human_eval['test']:
            method_name = re.findall('def .*\(', a_test['prompt'])[0].replace('def ', '').replace('(', '')
            test = a_test['test'] + '\ncheck(' + method_name + ')'
            final_test_cases.append(test)
        return final_test_cases

    def get_final_test_cases_separated(self):
        final_test_cases = []
        for a_test in self.human_eval['test']:
            method_name = re.findall('def .*\(', a_test['prompt'])[0].replace('def ', '').replace('(', '')
            test_cases = a_test['test'].split('assert candidate')[1:]
            for index, tt in enumerate(test_cases):
                tt.replace('candidate', method_name).replace("\\n", "").replace('\\t', '').replace("    ", "")
                test_cases[index] = "assert candidate" + tt
            # test = a_test['test'] + '\ncheck(' + method_name + ')'
            final_test_cases.append(test_cases)
        return final_test_cases

    def load_cod_eval(self):
        code_eval_metric = load("code_eval")
        return code_eval_metric