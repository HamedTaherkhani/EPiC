from transformers import pipeline
import torch
from openai import OpenAI
import os
import json
from humaneval_loader import HumanEvalLoader
from MBPPLoader import MBPPLoader
from utils import run_genetic_algorithm_gensim_
from BigCodeLoader import BigCodeLoader

class FireworksExperiments(object):
    def __init__(self):
        super(FireworksExperiments, self).__init__()

    def run_experiment_gensim(self,experiment_to_run, instances=None, population_size=5, dataset_choice=1, seed=137, mutation_tool=1) -> int:
        '''
        :param seed:
        :param first_generation_openai:
        :param instances:
        :param with_original_testcases: if true the intermediate evaluation is executed on the original test cases.
         if false the intermediate evaluation is executed on the generated test cases.
        :param population_size:
        :param dataset_choice: 1 = humaneval, 2 = mbpp, 3=BigcodeHard
        :param mutation_tool: 1 = local, 2 = LLM
        :return:

        '''

        model_name = 'deepseek-v3'
        if dataset_choice == 1:
            human_eval_loader = HumanEvalLoader(instances)
            human_eval = human_eval_loader.get_human_eval()
            final_test_cases = human_eval_loader.get_final_test_cases()
            # generated_testcases = get_testcases()
            generated_testcases = human_eval_loader.get_generated_test_cases_deepseek()
            dataset = [hh['prompt'] for hh in human_eval['test']]
            number_of_tests = len(dataset)
        elif dataset_choice == 2:
            mbpp_loader = MBPPLoader()
            final_test_cases = mbpp_loader.get_tests()
            generated_testcases = mbpp_loader.get_generated_testcases_deepseek()
            dataset = mbpp_loader.get_prompts()
            number_of_tests = len(dataset)
        else:
            loader = BigCodeLoader(hard=1)
            dataset = loader.get_prompts()
            number_of_tests = len(dataset)
            generated_testcases = loader.get_generated_tests_deepseek()
            final_test_cases = loader.get_tests()

        # dataset = dataset[42:45]
        # final_test_cases = final_test_cases[42:45]
        # generated_testcases = generated_testcases[42:45]
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
                                                   model_to_test=5,
                                                   # gpt_client=gpt_client,
                                                   population_size=population_size,
                                                   dataset_choice=dataset_choice,
                                                   seed=seed,
                                                   mutation_tool=mutation_tool,
                                                   model_name=model_name,
                                                   experiment_to_run=experiment_to_run)
        return final_pass
