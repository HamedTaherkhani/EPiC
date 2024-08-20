from transformers import pipeline
import torch
from openai import OpenAI
import os
import json
from humaneval_loader import HumanEvalLoader
from MBPPLoader import MBPPLoader
from utils import run_genetic_algorithm_gensim_


class GPTRunner:
    def __init__(self):
        super(GPTRunner, self).__init__()

    def run_experiment_gensim(self, instances=None, population_size=5, dataset_choice=1, seed=137, mutation_tool=1) -> int:
        '''
        :param seed:
        :param first_generation_openai:
        :param instances:
        :param with_original_testcases: if true the intermediate evaluation is executed on the original test cases.
         if false the intermediate evaluation is executed on the generated test cases.
        :param population_size:
        :param dataset_choice: 1 = humaneval, 2 = mbpp
        :param mutation_tool: 1 = local, 2 = LLM
        :return:

        '''

        key = os.getenv('openai_key')
        gpt_client = OpenAI(api_key=key)
        if dataset_choice == 1:
            human_eval_loader = HumanEvalLoader(instances)
            human_eval = human_eval_loader.get_human_eval()
            final_test_cases = human_eval_loader.get_final_test_cases()
            # generated_testcases = get_testcases()
            generated_testcases = human_eval_loader.get_generated_test_cases()
            dataset = [hh['prompt'] for hh in human_eval['test']]
            number_of_tests = len(dataset)
        else:
            mbpp_loader = MBPPLoader()
            final_test_cases = mbpp_loader.get_tests()
            generated_testcases = mbpp_loader.get_generated_testcases()
            dataset = mbpp_loader.get_prompts()
            number_of_tests = len(dataset)
        print(len(dataset))
        print(len(final_test_cases))
        print(len(generated_testcases))
        # print('final test cases **********************')
        # print(final_test_cases)
        final_pass = run_genetic_algorithm_gensim_(codeLLama_tokenizer=None,
                                                   codeLLama_model=None,
                                                   magic_coder=None, final_test_cases=final_test_cases,
                                                   generated_testcases=generated_testcases, dataset=dataset,
                                                   number_of_tests=number_of_tests,
                                                   model_to_test=2,
                                                   gpt_client=gpt_client,
                                                   population_size=population_size,
                                                   dataset_choice=dataset_choice,
                                                   seed=seed,
                                                   mutation_tool=mutation_tool)
        return final_pass
