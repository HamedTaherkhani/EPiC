import time
import os
import json
from dotenv import load_dotenv

load_dotenv()
def generate_first_population_openai():
    a = time.time()
    from evaluate import load
    code_eval_metric = load("code_eval")
    from datasets import load_dataset
    human_eval = load_dataset("openai_humaneval")
    from tqdm import tqdm
    code_eval_metric = load("code_eval")
    human_eval = load_dataset("openai_humaneval")
    from openai import OpenAI

    key = os.getenv('openai_key')
    client = OpenAI(api_key=key)
    gpt4_prompts = []
    for instance in tqdm(human_eval['test']):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user",
                       "content": "Refine the given prompt by enhancing its description to ensure clarity and comprehension for sophisticated language models. Maintain the original structure while elaborating on the details and ensuring it's easily understandable for advanced AI models. Use at most 500 tokens.\n" +
                                  instance['prompt']}],
            temperature=0.7,
            max_tokens=550,
            n=10
        )
        gpt4_prompts.append(response.choices)
        time.sleep(10)

    gpt_prompts_list = []
    for ins in gpt4_prompts:
        gpt_prompts_list.append([an_item.message.content for an_item in ins])

    print(gpt_prompts_list)
    print('total time:', time.time() - a)

def produce_testcases_for_mbpp():
    print('generating testcases openai for mbpp')
    from MBPPLoader import MBPPLoader
    import os
    from dotenv import load_dotenv

    load_dotenv()
    mbpp_random_instances = json.loads(os.getenv('mbpp_random_instances'))
    mbpp = MBPPLoader(mbpp_random_instances)
    prompts = mbpp.get_prompts()
    from evaluate import load
    from tqdm import tqdm
    from openai import OpenAI

    key = os.getenv('openai_key')
    openai_model = os.getenv("openai_model")
    print(openai_model)
    client = OpenAI(api_key=key)
    gpt4_prompts = []
    time1 = time.time()
    for idx, instance in tqdm(enumerate(prompts[:10])):
        # if idx == 5:
        #    break
        response = client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user",
                       "content": "Please write 10 valid test cases for this Python function. Verify all the testcases to ensure they are valid. Do not write the code for the function. Just write all the assertions inside another Python function. Don't write the code in a class; just a simple Python function. And do not include any descriptions or comments:\n" +
                                  instance}],
            temperature=0.0,
            max_tokens=600,
            n=1
        )
        gpt4_prompts.append(response.choices)

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
    print('total time')
    print(time.time() - time1)

if __name__ == '__main__':
    produce_testcases_for_mbpp()