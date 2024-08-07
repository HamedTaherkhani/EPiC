from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from chat_gpt_prompts import get_initial_processed_gpt_prompts
from humaneval_loader import HumanEvalLoader
from utils import run_genetic_algorithm, run_genetic_algorithm_gensim_, run_genetic_algorithm_gensim
from openai import OpenAI
import os
class CodellamaExperiments:
    def __init__(self):
        super(CodellamaExperiments, self).__init__()

    def load_codellama(self):
        ## load codellama in 4 b
        model_id = "codellama/CodeLlama-7b-instruct-hf"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            load_in_8bit_fp32_cpu_offload=True
        )

        codeLLama_tokenizer = AutoTokenizer.from_pretrained(model_id)
        codeLLama_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map='cuda:0',
        )
        return codeLLama_tokenizer, codeLLama_model
        ## load codellama original
        # codeLLama_tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-instruct-hf",
        #                                                          device_map='auto')
        # codeLLama_model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-7b-instruct-hf", device_map='auto')
        # codeLLama_model = codeLLama_model.to('cuda:0')

    def get_first_population(self, gpt_prompts, human_eval):
        base_prompts_re_codemagic = []
        for idx, base_prompts in enumerate(gpt_prompts):
            a = base_prompts[0:4]
            b = [human_eval['test'][idx]['prompt']]
            b.extend(a)
            base_prompts_re_codemagic.append(b)
        return base_prompts_re_codemagic

    def run_experiment(self, instances=None):

        codeLLama_tokenizer, codeLLama_model = self.load_codellama()
        gpt_prompts = get_initial_processed_gpt_prompts()
        human_eval_loader = HumanEvalLoader(instances)
        human_eval = human_eval_loader.get_human_eval()
        final_test_cases = human_eval_loader.get_final_test_cases()
        base_prompts_re_codemagic = self.get_first_population(gpt_prompts, human_eval)
        generated_testcases = human_eval_loader.get_generated_test_cases()
        if instances is not None:
            if len(instances) != 0:
                final_test_cases = [final_test_cases[i] for i in instances]
                base_prompts_re_codemagic = [base_prompts_re_codemagic[i] for i in instances]
                generated_testcases = [generated_testcases[i] for i in instances]

        run_genetic_algorithm(base_prompts_re=base_prompts_re_codemagic, codeLLama_tokenizer=codeLLama_tokenizer, codeLLama_model=codeLLama_model,
                              magic_coder=None, final_test_cases=final_test_cases,
                              generated_testcases=generated_testcases, human_eval=human_eval, number_of_tests=164, model_to_test=0)

    def run_experiments_gensim(self, instances=None, population_size=5, dataset_choice=1):
        codeLLama_tokenizer, codeLLama_model = self.load_codellama()
        human_eval_loader = HumanEvalLoader(instances)
        human_eval = human_eval_loader.get_human_eval()
        final_test_cases = human_eval_loader.get_final_test_cases()
        generated_testcases = human_eval_loader.get_generated_test_cases()
        dataset = [hh['prompt'] for hh in human_eval['test']]
        number_of_tests = len(dataset)

        key = os.getenv('openai_key')
        gpt_client = OpenAI(api_key=key)

        run_genetic_algorithm_gensim_(codeLLama_tokenizer=codeLLama_tokenizer,
                                      codeLLama_model=codeLLama_model,
                                      magic_coder=None, final_test_cases=final_test_cases,
                                      generated_testcases=generated_testcases, dataset=dataset,
                                      number_of_tests=number_of_tests,
                                      model_to_test=0,
                                      gpt_client=gpt_client,
                                      population_size=population_size,
                                      dataset_choice=dataset_choice)
