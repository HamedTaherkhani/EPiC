def generate_first_population_openai():
    from evaluate import load
    code_eval_metric = load("code_eval")
    from datasets import load_dataset
    human_eval = load_dataset("openai_humaneval")
    from tqdm import tqdm
    code_eval_metric = load("code_eval")
    human_eval = load_dataset("openai_humaneval")
    import time
    from openai import OpenAI

    key = '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'
    client = OpenAI(api_key=key)
    gpt4_prompts = []
    for instance in tqdm(human_eval['test']):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user",
                       "content": "Refine the given prompt by enhancing its description to ensure clarity and comprehension for sophisticated language models. Maintain the original structure while elaborating on the details and ensuring it's easily understandable for advanced AI models.\n" +
                                  instance['prompt']}],
            temperature=0.7,
            max_tokens=400,
            n=5
        )
        gpt4_prompts.append(response.choices)
        time.sleep(15)

    gpt_prompts_list = []
    for ins in gpt4_prompts:
        gpt_prompts_list.append([an_item.message.content for an_item in ins])

    print(gpt_prompts_list)

def produce_testcases():
    from evaluate import load
    code_eval_metric = load("code_eval")
    from datasets import load_dataset
    human_eval = load_dataset("openai_humaneval")
    import time
    from openai import OpenAI
    from tqdm import tqdm
    key = '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'
    client = OpenAI(api_key=key)
    gpt4_prompts = []
    for idx, instance in tqdm(enumerate(human_eval['test'])):
        # if idx == 2:
        #   break
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user",
                       "content": "Please write 10 test cases for this Python function. Do not write the code for the function. Just write all the assertions inside another Python function. Don't write the code in a class; just a simple Python function. And do not include any descriptions or comments:\n" +
                                  instance['prompt']}],
            temperature=0.0,
            max_tokens=600,
            n=1
        )
        gpt4_prompts.append(response.choices)
        time.sleep(15)
    produced_test_cases = []
    for test in gpt4_prompts:
        produced_test_cases.append(test[0].message.content)

    processed_test_cases = []
    for test in produced_test_cases:
        tests = []
        temp = test.split('assert')
        for a in temp[1:]:
            tests.append('assert ' + a.replace('\n', '').replace('```', ''))
        processed_test_cases.append(tests)
        # print(tests)
        # break
    print(processed_test_cases)
