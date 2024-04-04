from transformers import pipeline
import torch
from openai import OpenAI
import os
import json
from chat_gpt_prompts import get_initial_processed_gpt_prompts
from chat_gpt_prompts_distilled import refactor_prompts,get_gpt_prompts_distilled
from chat_gpt_prompts_distilled_ten_generation import refactor_prompts, get_gpt_prompts_distilled_ten
from gensim_prompts import get_gensim_prompts
from humaneval_loader import HumanEvalLoader
from chat_gpt_generated_testcases import get_testcases
from utils import run_genetic_algorithm, run_genetic_algorithm_gensim


class GPTRunner:
    def __init__(self):
        super(GPTRunner, self).__init__()

    def get_first_population(self, gpt_prompts, human_eval, population_size):
        base_prompts_re_codemagic = []
        for idx, base_prompts in enumerate(gpt_prompts):
            a = base_prompts[0:population_size]
            b = [human_eval['test'][idx]['prompt']]
            b.extend(a)
            base_prompts_re_codemagic.append(b)
        return base_prompts_re_codemagic

    def run_experiment_gensim(self, first_generation_openai=False, instances=None, with_original_testcases=False, population_size=5):

        key = os.getenv('openai_key')
        gpt_client = OpenAI(api_key=key)

        human_eval_loader = HumanEvalLoader()
        human_eval = human_eval_loader.get_human_eval()
        final_test_cases = human_eval_loader.get_final_test_cases()
        generated_testcases = get_testcases()

        if first_generation_openai:
            if population_size == 5:
                first_generation_prompts_refactored = self.get_first_population(get_gpt_prompts_distilled(), human_eval, population_size=5)
                first_generation_prompts_refactored = refactor_prompts(first_generation_prompts_refactored)
            elif population_size == 10:
                first_generation_prompts_refactored = self.get_first_population(get_gpt_prompts_distilled_ten(), human_eval,
                                                                                population_size=10)
                first_generation_prompts_refactored = refactor_prompts(first_generation_prompts_refactored)
        else:
            first_generation_prompts_refactored = get_gensim_prompts()

        if instances is not None:
            if len(instances) != 0:
                final_test_cases = [final_test_cases[i] for i in instances]
                first_generation_prompts_refactored = [first_generation_prompts_refactored[i] for i in instances]
                generated_testcases = [generated_testcases[i] for i in instances]

        run_genetic_algorithm_gensim(base_prompts_re=first_generation_prompts_refactored, codeLLama_tokenizer=None,
                                     codeLLama_model=None,
                                     magic_coder=None, final_test_cases=final_test_cases,
                                     generated_testcases=generated_testcases, human_eval=human_eval,
                                     number_of_tests=164,
                                     model_to_test=2, with_original_testcases=with_original_testcases,
                                     gpt_client=gpt_client,
                                     population_size=population_size)


