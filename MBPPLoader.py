import re
class MBPPLoader:
    def __init__(self, mbpp_random_instances):
        from datasets import load_dataset
        self.dataset = []
        dataset_full = load_dataset("mbpp")
        for key in dataset_full.keys():
            for item in dataset_full[key]:
                self.dataset.append(item)
        if mbpp_random_instances is not None:
            self.dataset = [dd for index, dd in enumerate(self.dataset) if index in mbpp_random_instances]
        self.prompts = [a['text'] for a in self.dataset]
        self.tests = ["\n".join(a['test_list'] + a['challenge_test_list']) for a in self.dataset]
        # self.funcname = [a['test_list'][0].split("(")[0].replace('assert', '').replace(' ', '') for a in self.dataset]
        self.prompts_ = []
        # print(self.dataset)
        for index, a in enumerate(self.dataset):
            entry = re.findall(r'def .*\(.*\)', a['code'])
            if len(entry) == 1:
                entry = entry[0]
            elif len(entry) > 1:
                entry = entry[-1]
                # print(index)

            ##correct func name
            to_be_replaces = entry.split("(")[0].replace("def ", '')
            # print(to_be_replaces)
            func_name = a['test_list'][0].split("(")[0].replace('assert', '').replace(' ', '')
            entry = entry.replace(to_be_replaces, func_name)

            entry += ":\n  \"\"\"" + a['text'] + "\"\"\""
            self.prompts_.append(entry)

    def get_prompts(self):
        return self.prompts_

    def get_tests(self):
        return self.tests

    def get_dataset(self):
        return self.dataset
