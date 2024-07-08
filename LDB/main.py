if __name__ == '__main__':
    ## generating mbpp with our testcases
    from MBPPLoader import MBPPLoader
    import json

    mbpp = MBPPLoader()
    dataset = mbpp.get_dataset()
    prompts = mbpp.get_prompts()
    func_names = mbpp.get_func_names()
    generated_tests = mbpp.get_generated_testcases()
    original_tests = mbpp.get_tests()
    mbpp_probs = []
    for idx, entry in enumerate(dataset):
        # print(entry)
        func_name = func_names[idx]
        # print(idx)
        # print(func_name)
        tests = '\ndef check(candidate):\n    ' + '\n    '.join(entry['test_list']) + f'\ncheck({func_name})'
        try:
            mbpp_probs.append({
                'task_id': "MBPP/" + str(entry['task_id']),
                'prompt': prompts[idx],
                'entry_point': func_name,
                'test': tests,
                'given_tests': [generated_tests[idx][0]],
            })
        except Exception as e:
            print(idx)
            print(func_name)
            mbpp_probs.append({
                'task_id': "MBPP/" + str(entry['task_id']),
                'prompt': prompts[idx],
                'entry_point': func_name,
                'test': tests,
                'given_tests': [entry['test_list'][0]],
            })
    with open('mbpp_with_generated_tests.jsonl', 'w') as outfile:
        for idx, entry in enumerate(mbpp_probs):
            json.dump(entry, outfile)
            outfile.write('\n')