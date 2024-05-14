from humaneval_loader import HumanEvalLoader
from LanguageAgentTreeSearch.programming.generators import generator_factory, model_factory
import time
import pickle
from tqdm import tqdm

if __name__ == "__main__":
    gen = generator_factory("python")
    human_eval = HumanEvalLoader(None).get_human_eval()['test']
    test_model = model_factory("gpt-4-0125-preview")
    number_of_tests = 10
    human_eval_testcases = []
    time1 = time.time()
    for idx,item in tqdm(enumerate(human_eval)):
        tests_i = gen.internal_tests(item["prompt"], test_model, number_of_tests)
        human_eval_testcases.append(tests_i)
    time2 = time.time()
    print(human_eval_testcases)
    with open('testcases/humaneval_generated_testcases', 'wb') as fp:
        pickle.dump(human_eval_testcases, fp)
    print("total_time:", time2 - time1)
