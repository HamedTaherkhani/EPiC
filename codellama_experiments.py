from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from chat_gpt_prompts import get_initial_processed_gpt_prompts
from humaneval_loader import HumanEvalLoader
from chat_gpt_generated_testcases import get_testcases
from utils import run_genetic_algorithm, run_genetic_algorithm_gensim
from gensim_prompts import get_gensim_prompts

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
        codellama7_errors = [[1, 'failed: '], [6, "failed: name 'groupby' is not defined"], [5, 'failed: '],
                             [8, "failed: name 'reduce' is not defined"], [10, 'failed: '], [11, 'failed: '],
                             [17, 'failed: '], [19, 'failed: '], [20, 'failed: '], [24, 'failed: '], [26, 'failed: '],
                             [32, "failed: poly() missing 1 required positional argument: 'x'"],
                             [33, "failed: expected ':' (<string>, line 23)"], [38, 'failed: '], [37, 'failed: '],
                             [36, 'failed: '], [39, "failed: name 'is_prime' is not defined"], [41, 'failed: '], [46,
                                                                                                                  'failed: unterminated triple-quoted string literal (detected at line 17) (<string>, line 4)'],
                             [50, 'failed: '], [51,
                                                'failed: unterminated triple-quoted string literal (detected at line 24) (<string>, line 4)'],
                             [54, 'failed: '], [57, 'failed: '], [59, "failed: name 'is_prime' is not defined"],
                             [62, 'failed: '], [63,
                                                'failed: unterminated triple-quoted string literal (detected at line 20) (<string>, line 4)'],
                             [64, 'failed: Test 4'], [65, 'failed: '],
                             [68, "failed: '(' was never closed (<string>, line 51)"],
                             [67, "failed: invalid literal for int() with base 10: 'and'"],
                             [69, "failed: '[' was never closed (<string>, line 25)"],
                             [71, 'failed: This prints if this assert fails 1 (good for debugging!)'], [70, 'failed: '],
                             [73, 'failed: '], [77, "failed: unsupported operand type(s) for %: 'complex' and 'int'"],
                             [75, "failed: name 'is_prime' is not defined"],
                             [76, 'failed: This prints if this assert fails 2 (also good for debugging!)'],
                             [79, 'failed: '],
                             [81, 'failed: unterminated string literal (detected at line 39) (<string>, line 39)'],
                             [78, 'failed: First test error: 2'], [80, 'failed: aabb'], [82, 'failed: '],
                             [84, 'failed: Error'], [85, 'failed: '], [86, 'failed: '],
                             [89, 'failed: This prints if this assert fails 1 (good for debugging!)'],
                             [88, 'failed: Error'], [87, 'failed: '], [90, 'failed: '],
                             [92, 'failed: This prints if this assert fails 1 (good for debugging!)'],
                             [93, 'failed: This prints if this assert fails 1 (good for debugging!)'],
                             [91, 'failed: Test 2'], [94, "failed: name 'is_prime' is not defined"],
                             [96, 'failed: invalid syntax (<string>, line 29)'], [98,
                                                                                  'failed: unterminated triple-quoted string literal (detected at line 24) (<string>, line 3)'],
                             [99, 'failed: Test 2'], [101, 'failed: '], [100, 'failed: Test 3'], [102, 'failed: '],
                             [104, 'failed: not all arguments converted during string formatting'], [103, 'failed: '],
                             [105, 'failed: Error'], [106, "failed: name 'factorial' is not defined"],
                             [107, 'failed: '], [109, 'failed: invalid syntax (<string>, line 44)'],
                             [108, "failed: invalid literal for int() with base 10: '-'"], [110, 'failed: '],
                             [113, 'failed: Test 1'], [111, "failed: name 'Counter' is not defined"], [112,
                                                                                                       'failed: unterminated triple-quoted string literal (detected at line 23) (<string>, line 3)'],
                             [114, 'failed: This prints if this assert fails 1 (good for debugging!)'],
                             [115, 'failed: Error'], [117, 'failed: First test error: []'],
                             [118, 'failed: string index out of range'], [119, 'failed: '], [120, 'failed: '],
                             [122, 'failed: invalid syntax (<string>, line 33)'],
                             [124, "failed: '[' was never closed (<string>, line 33)"], [123,
                                                                                         'failed: unterminated triple-quoted string literal (detected at line 18) (<string>, line 3)'],
                             [125,
                              'failed: unterminated triple-quoted string literal (detected at line 22) (<string>, line 3)'],
                             [126, 'failed: This prints if this assert fails 5 (good for debugging!)'],
                             [127, 'failed: '], [129, 'failed: '], [128, "failed: name 'prod' is not defined"],
                             [131, "failed: name 'reduce' is not defined"], [130,
                                                                             'failed: unterminated triple-quoted string literal (detected at line 23) (<string>, line 3)'],
                             [132, 'failed: '],
                             [133, 'failed: This prints if this assert fails 1 (good for debugging!)'],
                             [134, 'failed: '], [135, 'failed: '], [138, 'failed: '],
                             [137, "failed: '>' not supported between instances of 'int' and 'str'"],
                             [140, 'failed: This prints if this assert fails 4 (good for debugging!)'], [139,
                                                                                                         'failed: unterminated triple-quoted string literal (detected at line 15) (<string>, line 3)'],
                             [141, "failed: '[' was never closed (<string>, line 24)"],
                             [143, "failed: name 'is_prime' is not defined"], [144, 'failed: test1'], [145, 'failed: '],
                             [146, 'failed: '], [147, 'failed: '], [148, 'failed: First test error: 4'],
                             [150, 'failed: '], [149, 'failed: '],
                             [151, 'failed: This prints if this assert fails 3 (good for debugging!)'], [152,
                                                                                                         'failed: unterminated triple-quoted string literal (detected at line 18) (<string>, line 3)'],
                             [153, 'failed: '], [156, 'failed: '], [158, 'failed: t1'], [159, 'failed: Error'],
                             [161, 'failed: '], [160, 'failed: '], [163, 'failed: Test 1']]
        codellama7_errors = [aa[0] for aa in codellama7_errors]
        gpt_base_prompts = [a_prompt[0] for a_prompt in gpt_prompts]
        base_prompts_re = []
        for idx, base_prompts in enumerate(gpt_prompts):
            if idx not in codellama7_errors:
                base_prompts_re.append([human_eval['test'][idx]['prompt']])
            else:
                base_prompts_re.append(base_prompts)
        return base_prompts_re

    def run_experiment(self, instances=None):

        codeLLama_tokenizer, codeLLama_model = self.load_codellama()
        gpt_prompts = get_initial_processed_gpt_prompts()
        human_eval_loader = HumanEvalLoader()
        human_eval = human_eval_loader.get_human_eval()
        final_test_cases = human_eval_loader.get_final_test_cases()
        base_prompts_re_codemagic = self.get_first_population(gpt_prompts, human_eval)
        generated_testcases = get_testcases()
        if instances is not None:
            if len(instances) != 0:
                final_test_cases = [final_test_cases[i] for i in instances]
                base_prompts_re_codemagic = [base_prompts_re_codemagic[i] for i in instances]
                generated_testcases = [generated_testcases[i] for i in instances]

        run_genetic_algorithm(base_prompts_re=base_prompts_re_codemagic, codeLLama_tokenizer=codeLLama_tokenizer, codeLLama_model=codeLLama_model,
                              magic_coder=None, final_test_cases=final_test_cases,
                              generated_testcases=generated_testcases, human_eval=human_eval, number_of_tests=164, model_to_test=0)

    def run_experiments_gensim(self, instances=None):
        codeLLama_tokenizer, codeLLama_model = self.load_codellama()
        first_generation_prompts_refactored = get_gensim_prompts()
        human_eval_loader = HumanEvalLoader()
        human_eval = human_eval_loader.get_human_eval()
        final_test_cases = human_eval_loader.get_final_test_cases()
        generated_testcases = get_testcases()
        run_genetic_algorithm_gensim(base_prompts_re=first_generation_prompts_refactored,
                                     codeLLama_tokenizer=codeLLama_tokenizer,
                                     codeLLama_model=codeLLama_model,
                                     magic_coder=None, final_test_cases=final_test_cases,
                                     generated_testcases=generated_testcases, human_eval=human_eval, number_of_tests=164,
                                     model_to_test=0)
