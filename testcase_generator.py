import os
from dotenv import load_dotenv
import json
load_dotenv()
from humaneval_loader import HumanEvalLoader
from LanguageAgentTreeSearch.programming.generators import generator_factory, model_factory
import time
import pickle
from tqdm import tqdm
from MBPPLoader import MBPPLoader

def generate_for_mbpp():
    mbpp_random_instances = json.loads(os.getenv('mbpp_random_instances'))
    print(mbpp_random_instances)
    mbpp_loader = MBPPLoader(mbpp_random_instances)
    prompts = mbpp_loader.get_prompts()

    gen = generator_factory("python")
    test_model = model_factory("gpt4o")
    number_of_tests = 10
    generated_tests = []
    time1 = time.time()
    for instance in prompts:
        tests_i = gen.internal_tests(instance, test_model, number_of_tests)
        print(tests_i)
        generated_tests.append(tests_i)
    with open('testcases/mbpp_generated_testcases', 'wb') as fp:
        pickle.dump(generated_tests, fp)
    print(len(generated_tests))
    time2 = time.time()
    print("total_time:", time2 - time1)

def generate_for_humaneval():
    gen = generator_factory("python")
    human_eval = HumanEvalLoader(None).get_human_eval()['test']
    test_model = model_factory("gpt-4-0125-preview")
    number_of_tests = 2
    human_eval_testcases = []
    time1 = time.time()
    for idx, item in tqdm(enumerate(human_eval)):
        tests_i = gen.internal_tests(item["prompt"], test_model, number_of_tests)
        human_eval_testcases.append(tests_i)
    time2 = time.time()
    print(human_eval_testcases)
    with open('testcases/humaneval_generated_testcases', 'wb') as fp:
        pickle.dump(human_eval_testcases, fp)
    print("total_time:", time2 - time1)



if __name__ == "__main__":
    generate_for_mbpp()
    #generate_for_humaneval()
