from transformers import pipeline
import torch
from openai import OpenAI
import os
import json
from chat_gpt_prompts import get_initial_processed_gpt_prompts
from chat_gpt_prompts_distilled import refactor_prompts,get_gpt_prompts_distilled
from chat_gpt_prompts_distilled_ten_generation import refactor_prompts, get_gpt_prompts_distilled_ten
from humaneval_loader import HumanEvalLoader
from MBPPLoader import MBPPLoader
# from chat_gpt_generated_testcases import get_testcases
from test_cases import get_testcases
from utils import run_genetic_algorithm, run_genetic_algorithm_gensim_, run_genetic_algorithm_gensim_v2


class GPTRunner:
    def __init__(self):
        super(GPTRunner, self).__init__()

    def run_experiment_gensim(self, instances=None, population_size=5, version=1, dataset_choice=1, seed=137) -> int:
        '''
        :param seed:
        :param first_generation_openai:
        :param instances:
        :param with_original_testcases: if true the intermediate evaluation is executed on the original test cases.
         if false the intermediate evaluation is executed on the generated test cases.
        :param population_size:
        :param version:
        :param dataset_choice: 1 = humaneval, 2 = mbpp
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
        if version == 1:
            final_pass = run_genetic_algorithm_gensim_(codeLLama_tokenizer=None,
                                                       codeLLama_model=None,
                                                       magic_coder=None, final_test_cases=final_test_cases,
                                                       generated_testcases=generated_testcases, dataset=dataset,
                                                       number_of_tests=number_of_tests,
                                                       model_to_test=2,
                                                       gpt_client=gpt_client,
                                                       population_size=population_size,
                                                       dataset_choice=dataset_choice,
                                                       seed=seed)
            return final_pass
        # else:
        #     run_genetic_algorithm_gensim_v2(base_prompts_re=first_generation_prompts_refactored,
        #                                     codeLLama_tokenizer=None,
        #                                     codeLLama_model=None,
        #                                     magic_coder=None, final_test_cases=final_test_cases,
        #                                     generated_testcases=generated_testcases, human_eval=human_eval,
        #                                     number_of_tests=164,
        #                                     model_to_test=2, with_original_testcases=with_original_testcases,
        #                                     gpt_client=gpt_client,
        #                                     population_size=population_size)
