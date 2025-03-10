
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI # Assumes you are using OpenAI's API; replace with your LLM provider
from MBPPLoader import MBPPLoader
from humaneval_loader import HumanEvalLoader
from evaluator import CodeEval
from tqdm import tqdm
from utils import generate_code_fireworks, generate_code_sonnet
IMPORT_HEADER = "from typing import *\nimport math\nfrom heapq import *\nimport itertools\nimport re\nimport typing\nimport heapq\n_str=str\nimport re\n"

# Function to generate code using an LLM
def generate_code_gpt(prompt,gpt_client, model='o3-mini-2025-01-31',):
    """
    Generate code from a given prompt using an LLM.
    """
    response = gpt_client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are an AI assistant that writes Python functions."},
                  {"role": "user", "content": prompt}],
        # temperature=temperature,
        # max_tokens=300  # Adjust token limit as needed
    )
    text = response.choices[0].message.content
    return IMPORT_HEADER + '\n' + text, response.usage

# Function to perform zero-shot prompting using the dataset loader
def one_shot_prompting(loader_type="mbpp", model='o3-mini-2025-01-31',):
    """
    Perform zero-shot prompting using either MBPP or HumanEval.
    """
    total_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }
    key = os.getenv('openai_key')
    gpt_client = OpenAI(api_key=key)
    if loader_type.lower() == "mbpp":
        loader = MBPPLoader()
        prompts = loader.get_prompts()
        final_tests = loader.get_tests()
    elif loader_type.lower() == "humaneval":
        loader = HumanEvalLoader()
        final_tests = loader.get_final_test_cases()
        prompts = [item['prompt'] for item in loader.get_human_eval()['test']]
    else:
        raise ValueError("Invalid loader type. Choose 'mbpp' or 'humaneval'.")

    # Limit to a specified number of samples
    # prompts = prompts[:10]
    # final_tests = final_tests[:10]
    generated_codes = []
    for prompt in tqdm(prompts):
        if model == "o3-mini-2025-01-31":
            generated_code, usage = generate_code_gpt(prompt=prompt, gpt_client=gpt_client)
        elif model == "deepseek-v3":
            generated_code, usage = generate_code_fireworks(model_name=model, prompt=prompt)
        elif model == 'claude-3-7-sonnet-20250219':
            generated_code, usage = generate_code_sonnet(model_name=model, prompt=prompt)
        else:
            print('no known model...')
            return
        total_usage['total_tokens'] += usage.total_tokens
        total_usage['completion_tokens'] += usage.completion_tokens
        total_usage['prompt_tokens'] += usage.prompt_tokens
        generated_codes.append(generated_code)
        # print(generated_code)
        # print(f"Generated Code:\n{generated_code}\n{'-'*80}")

    print(f'total usage: {total_usage}')
    fillings = [[ff] for ff in generated_codes]
    pass_at_k, results = CodeEval()._compute(references=final_tests, predictions=fillings,k=[1])
    print(f"Pass at k: {pass_at_k}")

    print(f"Results: {results}")
    return generated_codes

# Example usage
if __name__ == "__main__":
    import sys
    loader_type = "MBPP"  # Change to "humaneval" to use HumanEvalLoader
    model = "claude-3-7-sonnet-20250219"
    orig_stdout = sys.stdout
    file_name = f'output/one_shot_{loader_type}_{model}.text'
    f = open(file_name, 'w')
    sys.stdout = f
    generated_codes = one_shot_prompting(loader_type)
    f.close()
    # print("\nFinal Generated Codes:\n", generated_codes)
