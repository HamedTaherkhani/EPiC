from dotenv import load_dotenv

load_dotenv()
from humaneval_loader import HumanEvalLoader
from TestcaseGenerator.generators import generator_factory, model_factory
from TestcaseGenerator.generators.test_generator_llamaapi import generate_tests_llamaapi
from prompts import PY_TEST_GENERATION_FEW_SHOT_BigCodeBench, PY_TEST_GENERATION_CHAT_INSTRUCTION_BigCodeBench
from BigCodeLoader import BigCodeLoader
from anthropic._exceptions import OverloadedError
import time
import pickle
from tqdm import tqdm
from MBPPLoader import MBPPLoader


def generate_for_mbpp(model_name):
    mbpp_loader = MBPPLoader()
    prompts = mbpp_loader.get_prompts()

    gen = generator_factory("python")
    test_model = model_factory(model_name)
    number_of_tests = 5
    generated_tests = []
    time1 = time.time()
    print(len(prompts))
    for instance in tqdm(prompts):
        try:
            tests_i = gen.internal_tests(instance, test_model, number_of_tests)
            print(tests_i)
            generated_tests.append(tests_i)
            time.sleep(30)
        except OverloadedError as e:
            print('OverloadedError')
            print('*'*100)
            tests_i = []
            time.sleep(60)
            # tests_i = gen.internal_tests(instance, test_model, number_of_tests)
            print(tests_i)
            generated_tests.append(tests_i)

    with open(f'testcases/mbpp_generated_testcases_{model_name}', 'wb') as fp:
        pickle.dump(generated_tests, fp)
    print(len(generated_tests))
    time2 = time.time()
    print("total_time:", time2 - time1)


def generate_for_humaneval(model_name):
    gen = generator_factory("python")
    human_eval = HumanEvalLoader(None).get_human_eval()['test']
    test_model = model_factory(model_name)
    number_of_tests = 10
    human_eval_testcases = []
    time1 = time.time()
    for idx, item in tqdm(enumerate(human_eval)):
        tests_i = gen.internal_tests(item["prompt"], test_model, number_of_tests)
        human_eval_testcases.append(tests_i)
        print(tests_i)
    time2 = time.time()
    print(human_eval_testcases)
    with open(f'testcases/humaneval_generated_testcases_{model_name}', 'wb') as fp:
        pickle.dump(human_eval_testcases, fp)
    print("total_time:", time2 - time1)

def generate_for_humaneval_llama():
    human_eval = HumanEvalLoader(None).get_human_eval()['test']
    human_eval_testcases = []
    for idx, item in tqdm(enumerate(human_eval)):
        # print(item)
        try:
            tests =generate_tests_llamaapi(model_name="llama3.1-8b", func_sig=item["prompt"])
            human_eval_testcases.append(tests)
        except OverloadedError as e:
            time.sleep(60)
        print(tests)
    with open('testcases/humaneval_generated_testcases_llama3', 'wb') as fp:
        pickle.dump(human_eval_testcases, fp)



if __name__ == "__main__":
    # model_name = "deepseek-v3"
    model_name = "claude-3.7-sonnet"
    generate_for_mbpp(model_name=model_name)
    # generate_for_humaneval(model_name=model_name)
    # generate_for_humaneval_llama()


