from transformers import pipeline
import torch
from chat_gpt_prompts import get_initial_population_from_chat_gpt
from humaneval_loader import HumanEvalLoader
from chat_gpt_generated_testcases import get_testcases
from utils import run_genetic_algorithm

import re


class MagicCoderRunner:
    def __init__(self):
        super(MagicCoderRunner, self).__init__()

    def load_magiccoder(self):
        import os
        os.environ['TRANSFORMERS_CACHE'] = '/home/hamedth/projects/def-hemmati-ac/hamedth/hugging_face'

        MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

        @@ Instruction
        {instruction}

        @@ Response
        """

        instruction = "def add(a,b):"

        prompt = MAGICODER_PROMPT.format(instruction=instruction)
        magic_coder = pipeline(
            model="ise-uiuc/Magicoder-S-DS-6.7B",
            task="text-generation",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        result = magic_coder(prompt, max_length=1024, num_return_sequences=1, temperature=0.0)
        return magic_coder
        print(result[0]["generated_text"])

    def get_first_population(self, gpt_prompts, human_eval):
        codemagic_errors = [1, 10, 15, 19, 26, 32, 39, 38, 41, 50, 54, 62, 64, 67, 76, 78, 77, 81, 84, 83, 86, 89, 90,
                            91, 93, 94, 95, 97, 96, 100, 104, 108, 109, 115, 119, 120, 125, 128, 126, 127, 129, 131,
                            130, 132, 134, 133, 135, 139, 140, 137, 142, 141, 143, 144, 146, 145, 150, 155, 160, 162,
                            163]

        base_prompts_re_codemagic = []
        for idx, base_prompts in enumerate(gpt_prompts):
            if idx not in codemagic_errors:
                base_prompts_re_codemagic.append([human_eval['test'][idx]['prompt']])
            else:
                base_prompts_re_codemagic.append(base_prompts)
        return base_prompts_re_codemagic

    def run_experiment(self):
        import os
        os.environ['TRANSFORMERS_CACHE'] = '/home/hamedth/projects/def-hemmati-ac/hamedth/hugging_face'
        magic_coder = self.load_magiccoder()
        gpt_prompts = get_initial_population_from_chat_gpt()
        human_eval_loader = HumanEvalLoader()
        human_eval = human_eval_loader.get_human_eval()
        final_test_cases = human_eval_loader.get_final_test_cases()
        base_prompts_re_codemagic = self.get_first_population(gpt_prompts, human_eval)
        generated_testcases = get_testcases()
        run_genetic_algorithm(base_prompts_re=base_prompts_re_codemagic, codeLLama_tokenizer=None, codeLLama_model=None,
                              magic_coder=magic_coder, final_test_cases=final_test_cases,
                              generated_testcases=generated_testcases, human_eval=human_eval, number_of_tests=40)
