import re
import pickle
class MBPPLoader:
    def __init__(self):
        from datasets import load_dataset
        self.dataset = []
        dataset_full = load_dataset("evalplus/mbppplus")['test']
        self.dataset = dataset_full
        self.prompts = [a['prompt'] for a in self.dataset]
        self.tests = [a['test'] for a in self.dataset]
        self.func_names = []
        # self.funcname = [a['test_list'][0].split("(")[0].replace('assert', '').replace(' ', '') for a in self.dataset]
        self.prompts_ = []
        for index, a in enumerate(self.dataset):
            entry = re.findall(r'def .*\(.*\)', a['code'])
            if len(entry) == 1:
                entry = entry[0]
            elif len(entry) > 1:
                entry = entry[-1]
                # print(index)
            entry += ":\n  \"\"\"" + a['prompt'] + "\"\"\""
            self.prompts_.append(entry)

    def get_generated_testcases_o3mini(self):
        import os
        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
        rel_path = "testcases/mbpp_generated_testcases_o3-mini"
        abs_file_path = os.path.join(script_dir, rel_path)
        with open(abs_file_path, 'rb') as fp:
            item_list = pickle.load(fp)
        def filter_list(l):
            temp = []
            for i in l:
                if 'assert False' in i or 'assert True' in i:
                    continue
                temp.append(i)
            return temp[:4]
        item_list = map(filter_list, item_list)
        return list(item_list)


    def get_generated_testcases_claude(self):
        import os
        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
        rel_path = "testcases/mbpp_generated_testcases_claude-3.7-sonnet"
        abs_file_path = os.path.join(script_dir, rel_path)
        with open(abs_file_path, 'rb') as fp:
            item_list = pickle.load(fp)
        def filter_list(l):
            temp = []
            for i in l:
                if 'assert False' in i or 'assert True' in i:
                    continue
                temp.append(i)
            return temp[:4]
        item_list = map(filter_list, item_list)
        return list(item_list)

    def get_generated_testcases_deepseek(self):
        import os
        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
        rel_path = "testcases/mbpp_generated_testcases_deepseek-v3"
        abs_file_path = os.path.join(script_dir, rel_path)
        with open(abs_file_path, 'rb') as fp:
            item_list = pickle.load(fp)
        def filter_list(l):
            temp = []
            for i in l:
                if 'assert False' in i or 'assert True' in i:
                    continue
                temp.append(i)
            return temp[:4]
        item_list = map(filter_list, item_list)
        return list(item_list)


    def get_generated_testcases(self):
        import os
        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
        rel_path = "testcases/mbpp_generated_testcases"
        abs_file_path = os.path.join(script_dir, rel_path)
        with open(abs_file_path, 'rb') as fp:
            item_list = pickle.load(fp)
        def filter_list(l):
            temp = []
            for i in l:
                if 'assert False' in i or 'assert True' in i:
                    continue
                temp.append(i)
            return temp[:10]
        item_list = map(filter_list, item_list)
        return list(item_list)

    def get_prompts(self):
        return self.prompts_

    def get_tests(self):
        return self.tests

    def get_func_names(self):
        return self.func_names

    def get_dataset(self):
        return self.dataset
