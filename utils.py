import json

import requests
import re
import pickle
from tqdm import tqdm
from evaluate import load
import time
import random
import numpy as np
import torch
import os
from generate_first_population import generate_first_population_for_instance
from TestcaseGenerator.executors import executor_factory
from evaluator import CodeEval
import anthropic
import openai
from function_executor import run_unit_tests_parallel, run_test_cases
from llamaapi import LlamaAPI
import os
import ast
from typing import List
from BigCodeLoader import BigCodeLoader
from dotenv import load_dotenv
load_dotenv()
key = os.getenv('openai_key')
llama_api_key = os.getenv('llama_api_key')

IMPORT_HEADER = "from typing import *\nimport math\nfrom heapq import *\nimport itertools\nimport re\nimport typing\nimport heapq\n_str=str\nimport re\n"


def generate_code_llamaapi(model_name, prompt):
    llama = LlamaAPI(llama_api_key)
    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }
    api_request_json = {
        "model": model_name,
        "max_tokens": 2000,
        'messages': [{"role": "system",
                     "content": "You are a python developer that implements the correct code based on the function description provided. You are given one or more functions to implement. Don't delete import statements in the code snippet. Use at most 1000 words."},
                    {"role": "user",
                     "content": prompt.replace(
                         "#SPECIAL_TOKEN", "")}],
    "stream": False,
    }
    try:
        response = llama.run(api_request_json)
        response = response.json()
    except Exception as e:
        print(e)
        return prompt, usage
    content = response['choices'][0]['message']['content']
    IMPORT_HEADER = "from typing import *\nimport math\nfrom heapq import *\nimport itertools\nimport re\nimport typing\nimport heapq\n_str=str\nimport re\n"
    ##process
    try:
        filling = IMPORT_HEADER + prompt + '\n' + content.split('```')[1].replace('python', '')
        usage = response['usage']
    except IndexError:
        filling = IMPORT_HEADER + prompt
    return filling, usage

def generate_code_fireworks(model_name, prompt):
    fire_work_key = os.getenv('fireworks_key')
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    payload = {
        "model": f"accounts/fireworks/models/{model_name}",
        "max_tokens": 8000,
        # "top_p": 1,
        # "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {fire_work_key}"
    }
    res = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    text = res.json()['choices'][0]['message']['content']
    usage = res.json()['usage']
    try:
        filling = IMPORT_HEADER + prompt + '\n' + text.split('```')[1].replace('python', '')
    except IndexError:
        filling = IMPORT_HEADER + text
        ###
    return filling, usage

def generate_code_sonnet(model_name, prompt):
    client = anthropic.Anthropic(api_key=os.getenv("anthropic_key"))
    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }
    try:
        time.sleep(1)
        message = client.messages.create(
            model=model_name,
            max_tokens=2000,
            temperature=0,
            system="You are an expert python developer",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Implement the right python implementation for this function. Import any necessary libraries and put the implementation between ```python and ``` tags" + prompt
                        }
                    ]
                }
            ]
        )
        text = message.content[0].text
        usage = {
            "prompt_tokens": message.usage.input_tokens,
            "completion_tokens": message.usage.output_tokens,
            "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
        }
        try:
            filling = IMPORT_HEADER + prompt + '\n' + text.split('```')[1].replace('python', '')
        except IndexError:
            filling = IMPORT_HEADER + text
            ###
        return filling, usage
    except Exception as e:
        time.sleep(10)
        print('Excetion in sonnet API')
        print(e)
        return prompt, usage


MUtation_llm = {
    1: 'Lama70b',
    2: 'Lama7b'
}
from openai import OpenAI

from multiprocessing import Pool
from itertools import repeat
from itertools import product
from gensimutils import mutate_sentence, mutate_prompt
def f(a_test, candidates):
    print(a_test)
    print(candidates)
    pass_at_k, results = code_eval_metric.compute(references=[a_test], predictions=candidates, k=[1])
    return pass_at_k


special_token = "#SPECIAL_TOKEN"
code_eval_metric = load("code_eval",timeout=1000)
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def choose_candidates(prompts_set, number=1):
    if number == 0:
        return []
    chosen_prompts = []
    try:
        for i in range(number):
            temp = random.choices(prompts_set, weights=(ss[1] for ss in prompts_set), k=1)[0]
            # prompts_set.remove(temp)
            chosen_prompts.append(temp)
    except ValueError:
        chosen_prompts = random.choices(prompts_set, k=number)
    return chosen_prompts


def process_prompt(res, a_candidate):
    f = res.find("# Explanation:")
    e = res.find("# End")
    p = a_candidate
    sp = p.find('def')
    comment = p[sp + 3:].find('"""')
    if comment == -1:
        comment = p[sp + 3:].find("'''")
    examples = p[sp + 3 + comment + 3:].find('>>>')
    if examples == -1:
        examples = p[sp + 3 + comment + 3:].find('"""')
        if examples == -1:
            examples = p[sp + 3 + comment + 3:].find("'''")
    llama_prompts_final = p.replace(p[sp + 3 + comment + 3:sp + 3 + comment + 3 + examples],
                                    res[f + len('# Explanation:'):e])
    if len(llama_prompts_final) > 1000:
        llama_prompts_final = llama_prompts_final[0:1000]
    return llama_prompts_final


def process_prompt2(res, a_candidate):
    f = res.find("Explanation")
    e = res.find("End")
    p = a_candidate
    sp = p.find('def')
    comment = p[sp + 3:].find(special_token)
    if comment == -1:
        comment = p[sp + 3:].find("'''")
    examples = p[sp + 3 + comment + 3:].find('>>>')
    if examples == -1:
        examples = p[sp + 3 + comment + 3:].find(special_token)
        if examples == -1:
            examples = p[sp + 3 + comment + 3:].find("'''")
    llama_prompts_final = p.replace(p[sp + 17 + comment + 3:sp + 3 + comment + 3 + examples],
                                    res[f + len('Explanation:'):e])
    if len(llama_prompts_final) > max_response_length + 1:
        llama_prompts_final = llama_prompts_final[0:max_response_length]
    return llama_prompts_final


def process_api_prompt(res, a_candidate):
    f = res.find("Explanation")
    e = res.find("End")
    res = res[f + 11:e]
    # if len(res) > max_response_length + 1:
    #     res = res[0:max_response_length]
    exp = [m.start() for m in re.finditer(special_token, a_candidate)]
    if len(exp) == 0:
        exp = [m.start() for m in re.finditer("'''", a_candidate)]
    if len(exp) == 1:
        exp.append(-1)
    elif len(exp) == 0:
        exp = [0, -1]
    final_prompt = a_candidate.replace(a_candidate[exp[0] + 17: exp[1] - 3], res)
    return final_prompt


# def process_prompt(res, a_candidate):

headers = {
    'Content-Type': 'text/plain'
}

def augment_promt(a_candidate):
    # print(a_candidate)
    prompt = a_candidate[0]
    code = a_candidate[2]
    feedback = a_candidate[3]
    feeds = ''
    for idx,ff in enumerate(feedback):
        feeds += f"""
#### Test {idx}:
{ff[2]}
### Feedback {idx}:
{ff[1]}
        """
    final_prompt = f"""
    Debug and fix the given code based on the provided test execution feedbacks.

### CODE:
{code}

{feeds}
### INSTRUCTIONS:
- Analyze the feedback carefully to identify the issues in the code.
- Fix all the errors and improve the code where necessary.
- Ensure all test cases pass successfully.
- Maintain the original logic and intent of the code as much as possible.
Return only the corrected code"""
    return final_prompt


def mutate_prompt_gpt(a_candidate, gpt_client, model_name):
    query2 = "You are a mutation tool. This is a python function and it's description. Please change the description by enhancing it's clarity and comprehension for sophisticated language models. Please put the changed description between #Explanation and #End. Use at most 1000 words."
    prompt = a_candidate[0]
    response = gpt_client.chat.completions.create(model=model_name,
                                                  messages=[
                                                      {"role": "user",
                                                       "content": query2 + '\n\n' + prompt}],
                                                  # temperature=0.8,
                                                  # max_tokens=750,
                                                  )
    usage = response.usage
    usage = {
        'completion_tokens': usage.completion_tokens,
        'prompt_tokens': usage.prompt_tokens,
        'total_tokens': usage.total_tokens,
    }
    return process_api_prompt(response.choices[0].message.content, prompt), usage

def mutate_prompt_gpt_v2(a_candidate, gpt_client, model_name):
    prompt = a_candidate[0]
    code = a_candidate[2]
    feedback = a_candidate[3]
    feeds = ''
    for idx, ff in enumerate(feedback):
        feeds += f"""
    #### Test {idx}:
    {ff[2]}
    ### Feedback {idx}:
    {ff[1]}
            """
    final_prompt = f"""
We have generated code based on the given prompt. Now, using both the generated code and the execution feedback from the tests, refine the original prompt to incorporate the insights from the test results. Ensure that the revised prompt provides clearer guidance to the LLM, improving its ability to generate the correct code.
Place the enhanced prompt (including function signature, input and output types) between three asterisks (***). Don't place anything else between three asterisks (***) just the enhanced prompt. Don't implement the code.

    ### Prompt:
    {prompt}
    
    ### CODE:
    {code}

    {feeds}"""
    # print(final_prompt)
    response = gpt_client.chat.completions.create(model=model_name,
                                                  messages=[{"role": "user",
                                                             "content": final_prompt}],
                                                  # temperature=0.8,
                                                  # max_tokens=750,
                                                  )
    response = response.choices[0].message.content
    match = re.search(r'\*\*\*(.*?)\*\*\*', response, re.DOTALL)
    return match.group(1).strip() if match else ""



def mutate_prompts_api(a_candidate, mutation_llm):
    query2 = "Here is a python function and it's description. Please Refine and elaborate the description by enhancing it's clarity and comprehension for sophisticated language models. Please put the refined description between. Use at most 400 words #Explanation and #End. \\n"
    url = "https://www.llama2.ai/api"
    prompt_changed = a_candidate.replace('\n', '\\n').replace("\"", '\\"')
    if mutation_llm == 1:
        payload = "{\"prompt\":\"[INST]Hello [/INST]\\n\",\"model\":\"meta/llama-2-70b-chat\",\"systemPrompt\":\"You are a helpful assistant.\",\"temperature\":0.5,\"topP\":0.9,\"maxTokens\":1000,\"image\":null,\"audio\":null}"
    elif mutation_llm == 2:
        payload = "{\"prompt\":\"[INST]Hello [/INST]\\n\",\"model\":\"meta/llama-2-7b-chat\",\"systemPrompt\":\"You are a helpful assistant.\",\"temperature\":0.5,\"topP\":0.9,\"maxTokens\":1000,\"image\":null,\"audio\":null}"
    else:
        raise Exception('Invalid mutation_llm')
    payload = payload.replace("Hello", query2 + prompt_changed)

    response = requests.request("POST", url, headers=headers, data=payload).text
    return process_api_prompt(response, a_candidate)


def crossover_prompts_api(cands, mutation_llm):
    two_candidates = choose_candidates(cands, 2)
    first_temp = [m.start() for m in re.finditer(special_token, two_candidates[0])]
    if len(first_temp) == 0:
        first_temp = [m.start() for m in re.finditer("'''", two_candidates[0])]
    if len(first_temp) == 1:
        first_temp.append(-1)
    elif len(first_temp) == 0:
        first_temp = [0, -1]

    second_temp = [m.start() for m in re.finditer(special_token, two_candidates[1])]
    if len(second_temp) == 0:
        second_temp = [m.start() for m in re.finditer("'''", two_candidates[1])]

    if len(second_temp) == 1:
        second_temp.append(-1)
    elif len(second_temp) == 0:
        second_temp = [0, -1]

    PROMPT = 'Merge the first explanation and the second explanation to have a new explanation for the function. Please put the new explanation after # Explanation: and before # End .\n' + 'first explanation:\n' + \
             two_candidates[0][first_temp[0] + 17: first_temp[1]] + 'second explanation:\n' + two_candidates[1][
                                                                                              second_temp[0] + 17:
                                                                                              second_temp[1]]
    url = "https://www.llama2.ai/api"
    prompt_changed = PROMPT.replace('\n', '\\n').replace("\"", '\\"')
    if mutation_llm == 1:
        payload = "{\"prompt\":\"[INST]Hello [/INST]\\n\",\"model\":\"meta/llama-2-70b-chat\",\"systemPrompt\":\"You are a helpful assistant.\",\"temperature\":0.5,\"topP\":0.9,\"maxTokens\":1000,\"image\":null,\"audio\":null}"
    elif mutation_llm == 2:
        payload = "{\"prompt\":\"[INST]Hello [/INST]\\n\",\"model\":\"meta/llama-2-7b-chat\",\"systemPrompt\":\"You are a helpful assistant.\",\"temperature\":0.5,\"topP\":0.9,\"maxTokens\":1000,\"image\":null,\"audio\":null}"
    payload = payload.replace("Hello", prompt_changed)
    response = requests.request("POST", url, headers=headers, data=payload).text
    return process_api_prompt(response, two_candidates[0])


def validate_prompt(prompt):
    if 'def' in prompt:
        return True
    return False


def process_the_code_magic_coder(fillings, human_eval):
    processed_fillings = []
    for index, a_fil in enumerate(fillings):
        method_names = re.findall('def .*\(', human_eval['test'][index]['prompt'])
        number_of_methods = len(method_names)
        try:
            if index in (106, 119):
                filling = a_fil
                processed_fillings.append(filling)
                continue
            a_fil = a_fil.replace('print', '#print')
            methods = a_fil.split('def ')
            if number_of_methods == 1:
                filling = methods[0] + 'def ' + methods[1]
            elif number_of_methods == 2:
                filling = methods[0] + 'def ' + methods[1] + 'def ' + methods[2]
            filling = filling.split('# Test cases')[0]
        except Exception as e:
            print(index)
            filling = a_fil
        processed_fillings.append(filling)
    return processed_fillings


def process_a_code_magic_coder(filling, index, human_eval):
    method_names = re.findall('def .*\(', human_eval['test'][index]['prompt'])
    number_of_methods = len(method_names)
    try:
        if index in (106, 119):
            return filling
        filling = filling.replace('print', '#print')
        methods = filling.split('def ')
        if number_of_methods == 1:
            filling = methods[0] + 'def ' + methods[1]
        elif number_of_methods == 2:
            filling = methods[0] + 'def ' + methods[1] + 'def ' + methods[2]
        filling = filling.split('# Test cases')[0]
    except Exception as e:
        print(index)
        return filling

    return filling


def evaluate_prompt(test_cases, prompt, codeLLama_tokenizer, codeLLama_model, magic_coder, human_eval, model_to_test=0, prompt_index=None):
    if model_to_test == 0:
        prompt = codeLLama_tokenizer(prompt, return_tensors="pt")["input_ids"].to('cuda:0')
        generated_id = codeLLama_model.generate(prompt.replace('#SPECIAL_TOKEN', ''), max_new_tokens=128)
        filling = codeLLama_tokenizer.batch_decode(generated_id, skip_special_tokens=True)[0]
        ## process
        try:
            aas = filling.split('def')
            filling = aas[0] + 'def' + aas[1]
        except Exception as e:
            # print(prompt)
            return 0, 0
    elif model_to_test == 1:
        filling = \
        magic_coder(prompt.replace('#SPECIAL_TOKEN', ''), max_length=512, num_return_sequences=1, temperature=0.0)[0][
            'generated_text']
        filling = process_a_code_magic_coder(filling, prompt_index, human_eval)
    ##
    candidate = [filling]
    candidates = [candidate]
    pass_at_k, results = code_eval_metric.compute(references=[test_cases], predictions=candidates, k=[1])
    return pass_at_k['pass@1']


def get_gpt_code_completion(gpt_client, prompt, model_name):
    counter = 0
    number_of_tries = 5
    usage = {
        'completion_tokens': 0,
        'prompt_tokens': 0,
        'total_tokens': 0,
    }
    if 'o3-mini' not in model_name:
        params = {
            'temperature': 0,
            'max_tokens': 1024,
        }
    else:
        params = {
            # 'max_completion_tokens': 2024,
        }
    while True:
        try:
            if counter == number_of_tries:
                print(f'code completion failed for prompt: {prompt}')
                return prompt, usage
            response = gpt_client.chat.completions.create(model=model_name,
                                                          messages=[{"role": "system",
                                                                     "content": "You are a python developer that implements the correct code based on the function description provided. You are given one or more functions to implement. Don't delete import statements in the code snippet. Use at most 2000 words. Import any necessary libraries used in the code."},
                                                                    {"role": "user",
                                                                     "content":  "You are a python developer that implements the correct code based on the function description provided. You are given one or more functions to implement. Don't delete import statements in the code snippet. Use at most 2000 words. Import any necessary libraries used in the code.\n\n" +prompt.replace(
                                                                         "#SPECIAL_TOKEN", "")}],
                                                          **params
                                                          )
            filling = response.choices[0].message.content
            usage = response.usage
            usage = {
                'completion_tokens': usage.completion_tokens,
                'prompt_tokens': usage.prompt_tokens,
                'total_tokens': usage.total_tokens,
            }
            break
        except openai.InternalServerError:
            print('Internal Server Error OpenAI, waiting 10 seconds...')
            time.sleep(10)
            counter += 1
    # print(filling)
    # print('-'*100)

    ##process
    try:
        filling = IMPORT_HEADER + prompt + '\n' + filling.split('```')[1].replace('python', '')
    except IndexError:
        filling = IMPORT_HEADER + filling
    ###
    return filling, usage

def evaluate_prompt_on_generated_prompts(generated_test_cases, prompt, codeLLama_tokenizer, codeLLama_model, magic_coder, human_eval, model_name,
                                         model_to_test=0, prompt_index=None, gpt_client=None, dataset_choice=1):
    usage ={
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
    # if not validate_prompt(
    #         prompt):
    #     return 0, 0, usage
    if model_to_test == 0:
        prompt = codeLLama_tokenizer(prompt.replace('#SPECIAL_TOKEN', ''), return_tensors="pt")["input_ids"].to(
            'cuda:0')
        generated_id = codeLLama_model.generate(prompt, max_new_tokens=128)
        filling = codeLLama_tokenizer.batch_decode(generated_id, skip_special_tokens=True)[0]
        ## process
        try:
            aas = filling.split('def')
            filling = aas[0] + 'def' + aas[1]
        except Exception as e:
            # print(prompt)
            return 0, 0, usage
    elif model_to_test == 1:
        filling = \
        magic_coder(prompt.replace('#SPECIAL_TOKEN', ''), max_length=512, num_return_sequences=1, do_sample=False)[0][
            'generated_text']
        filling = process_a_code_magic_coder(filling, prompt_index,human_eval)
    elif model_to_test == 2:
            filling, usage = get_gpt_code_completion(gpt_client, prompt, model_name)
    elif model_to_test == 3:
        filling, usage = generate_code_llamaapi(model_name=model_name, prompt=prompt)
        time.sleep(1)
    elif model_to_test == 4:
        filling, usage = generate_code_sonnet(model_name=model_name, prompt=prompt)
        time.sleep(1)
    elif model_to_test == 5:
        filling, usage = generate_code_fireworks(model_name=model_name, prompt=prompt)
    ##
    candidate = [filling]
    candidates = [candidate]

    pass_total = 0
    if dataset_choice == 3:
        # print('running unittest evaluation')
        res = run_unit_tests_parallel(code_str=filling, test_list=generated_test_cases)
        # for a in res:
        #     print(a[1])
        try:
            passat1 = len([an for an in res if an[0]==True]) / len(res)
            return passat1, filling, usage
        except ZeroDivisionError:
            return 1, filling, usage
    else:
        for a_test in generated_test_cases:
            # print(a_test)
            # print(filling)
            # print('here2')
            pass_at_k, results = code_eval_metric.compute(references=[a_test], predictions=candidates, k=[1])
            pass_total += pass_at_k['pass@1']
        try:
            return pass_total / len(generated_test_cases), filling, usage
        except ZeroDivisionError:
            return 1, filling, usage

    # with Pool() as p:
    #     results = p.starmap(f, zip(test_cases, repeat(candidates)))
    # print(results)
    # return sum(results) / len(results)


def run_final_evaluation(chosen_prompts, codeLLama_model, codeLLama_tokenizer, evaluations, final_test_cases,
                         human_eval, iteration, magic_coder, model_to_test, number_of_tests, passed_codes, time_test,model_name,dataset_choice, gpt_client=None):

    e = time.time()
    if iteration != 1000:
        if model_to_test == 0:
            fillings = []
            for index, a_token in tqdm(enumerate(chosen_prompts)):
                if not validate_prompt(
                        a_token):  ##this is because codeLLama_model has no max_new_tokens set and generates infinite output
                    filling = 'teeeeeeeeeeeeeeeeest'
                    fillings.append([filling])
                    continue
                if not passed_codes[index]:
                    prompt = codeLLama_tokenizer(a_token, return_tensors="pt")["input_ids"]  ##.to('cuda:0')
                    generated_id = codeLLama_model.generate(prompt, max_new_tokens=128)
                    filling = codeLLama_tokenizer.batch_decode(generated_id, skip_special_tokens=True)[0]
                    try:
                        aas = filling.split('def')
                        filling = aas[0] + 'def' + aas[1]
                    except IndexError:
                        filling = 'teeeeeeeeeeeeeeeeest'
                    fillings.append([filling])
                else:
                    fillings.append([passed_codes[index]])
        elif model_to_test == 1:
            fillings = []
            for index, a_token in tqdm(enumerate(chosen_prompts)):
                if not validate_prompt(
                        a_token):  ##this is because codeLLama_model has no max_new_tokens set and generates infinite output
                    filling = 'teeeeeeeeeeeeeeeeest'
                    fillings.append([filling])
                    continue
                if not passed_codes[index]:
                    filling = magic_coder(a_token.replace('#SPECIAL_TOKEN', ''), max_length=800, num_return_sequences=1,
                                          temperature=0.0)[0]['generated_text']
                    fillings.append(filling)
                else:
                    fillings.append(passed_codes[index])
            fillings = process_the_code_magic_coder(fillings, human_eval)
            fillings = [[fil] for fil in fillings]
        elif model_to_test == 2:
            fillings = []
            for index, a_token in tqdm(enumerate(chosen_prompts)):
                if not validate_prompt(
                        a_token):
                    filling = 'teeeeeeeeeeeeeeeeest'
                    fillings.append([filling])
                    continue
                if not passed_codes[index]:
                    filling, _ = get_gpt_code_completion(gpt_client, a_token, model_name)
                    fillings.append(filling)
                else:
                    fillings.append(passed_codes[index])
            fillings = [[fil] for fil in fillings]
        elif model_to_test == 3:
            fillings = []
            for index, a_token in tqdm(enumerate(chosen_prompts)):
                if not validate_prompt(
                        a_token):
                    filling = 'teeeeeeeeeeeeeeeeest'
                    fillings.append([filling])
                    continue
                if not passed_codes[index]:
                    filling, _ = generate_code_llamaapi(model_name=model_name, prompt=a_token)
                    fillings.append(filling)
                else:
                    fillings.append(passed_codes[index])
            fillings = [[fil] for fil in fillings]
        elif model_to_test in (4,5):
            fillings = []
            for index, a_token in tqdm(enumerate(chosen_prompts)):
                if not validate_prompt(
                        a_token):
                    filling = 'teeeeeeeeeeeeeeeeest'
                    fillings.append([filling])
                    continue
                if not passed_codes[index]:
                    if model_to_test == 4:
                        filling, _ = generate_code_sonnet(model_name=model_name, prompt=a_token)
                    else:
                        filling, _ = generate_code_fireworks(model_name=model_name, prompt=a_token)
                    fillings.append(filling)
                else:
                    fillings.append(passed_codes[index])
            fillings = [[fil] for fil in fillings]
        if dataset_choice == 3:
            print('please perform final evaluation on bigcodebench github...')
            return fillings, []
        errorrrs = []
        # print(fillings[2])
        # print(final_test_cases[2])
        # num_passed = 0
        # for idx_1,fil in enumerate(fillings):
        #     print(fil[0])
        #     for aaa in final_test_cases[idx_1]:
        #         print(aaa)
        #     print('*'*100)
        #     res = run_test_cases(fil[0],final_test_cases[idx_1],timeout=120)
        #     all_passed = all(res)
        #     num_passed += 1 if all_passed else 0
        # pass_at_k = num_passed / len(fillings)
        # pass_at_k, results = code_eval_metric.compute(references=final_test_cases[0:number_of_tests],
        #                                               predictions=fillings, k=[1])  ##here
        pass_at_k, results = CodeEval()._compute(references=final_test_cases[0:number_of_tests], predictions=fillings, k=[1])
        # for key, value in results.items():
        #     if value[0][1]['passed']:
        #         passed_codes[value[0][1]['task_id']] = fillings[value[0][1]['task_id']][0]
        evaluations.append((pass_at_k, results))
        print(pass_at_k)
        errors = []
        for index, result in results.items():
            if not result[0][1]['passed']:
                errors.append((result[0][1]['task_id'], result[0][1]['result']))
        errors_index = [err[0] for err in errors]
        print("errors *********************:  ", errors_index)
        print(results)
        # print('prompts:')
        # print(chosen_prompts)
        # print('fillings:')
        # print(fillings)
        time_test.append(time.time() - e)
        return fillings, errors_index
        # for key,item in results[1].items():
        #     if item[0][1]['passed']:
        #         passed_fillings[item[0][1]['task_id']] = fillings[item[0][1]['task_id']][0]


def print_time_measures(evaluations, number_of_supposed_passed_codes, start, time_evaluation, time_next_make_generation,
                        time_test, time_total_per_instance, usage):
    print('number_of_supposed_passed_codes')
    print(number_of_supposed_passed_codes)
    print('time_total_per_instance')
    print(time_total_per_instance)
    print('time_next_make_generation')
    print(time_next_make_generation)
    print('time_evaluation')
    print(time_evaluation)
    print('time_test')
    print(time_test)
    print('time_total')
    time_total = time.time() - start
    print(time_total)
    print(evaluations)
    # print('time_total_per_instance for every loop:')
    # print(np.sum(time_total_per_instance, axis=1) + time_test)
    print('total time:')
    print(time_total)
    print('Total time - final evaluations in loop')
    print(np.sum(np.sum(time_total_per_instance, axis=1)))
    print('total instance mean')
    print(np.mean(np.sum(time_total_per_instance, axis=0)))
    print(np.median(np.sum(time_total_per_instance, axis=0)))
    print('next generation make time:')
    print(np.sum(time_next_make_generation))
    print(np.sum(time_next_make_generation, axis=1))
    print('evaluation(code generation and running test cases)')
    print(np.sum(time_evaluation))
    print(np.sum(time_evaluation, axis=1))
    print(f'total usage: {usage}')


def run_genetic_algorithm(base_prompts_re, codeLLama_tokenizer, codeLLama_model, magic_coder, final_test_cases, generated_testcases, human_eval, number_of_tests=164, model_to_test=0, mutation_llm=1):
    all_generated_promts = []
    # all_generated_promts = []
    evaluations = []
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import warnings
    warnings.filterwarnings("ignore")
    iterations = 4
    run_evaluation_each_generation = True
    ## time management
    time_total_per_instance = []
    time_evaluation = []
    time_test = []
    time_next_make_generation = []
    number_of_supposed_passed_codes = []
    # if model_to_test == 1:
    #     base_prompts_re = base_prompts_re_codemagic.copy()

    passed_codes = [False for i in range(number_of_tests)]
    start = time.time()
    for iteration in tqdm(range(iterations)):
        torch.cuda.empty_cache()
        time_total_per_instance.append([])
        time_evaluation.append([])
        time_next_make_generation.append([])

        all_generated_promts.append(base_prompts_re.copy())
        number_of_supposed_passed_codes.append(0)
        for idx, a_prompt_set in tqdm(enumerate(base_prompts_re[0:number_of_tests])):  ##here
            print(idx)
            c = time.time()
            passed = False
            if len(a_prompt_set) == 1:
                time_total_per_instance[iteration].append(0)
                time_evaluation[iteration].append(0)
                time_next_make_generation[iteration].append(0)
                passed = True
                number_of_supposed_passed_codes[iteration] +=1
                continue
            else:
                candidates = []
                a = time.time()
                for single_prompt in a_prompt_set:
                    passed = False
                    passat10 = evaluate_prompt_on_generated_prompts(generated_test_cases=generated_testcases[idx][0:4],
                                                                    prompt=single_prompt, model_to_test=model_to_test,
                                                                    prompt_index=idx,
                                                                    codeLLama_tokenizer=codeLLama_tokenizer,
                                                                    codeLLama_model=codeLLama_model,
                                                                    magic_coder=magic_coder,
                                                                    human_eval=human_eval)

                    candidates.append([single_prompt, passat10])
                    if passat10 == 1:
                        base_prompts_re[idx] = [single_prompt]
                        print(
                            f'PAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAASED for idx {idx}')
                        passed = True
                        break
                if passed:
                    b = time.time()
                    time_evaluation[iteration].append(b-a)
                    time_total_per_instance[iteration].append(b-a)
                    time_next_make_generation[iteration].append(0)
                    continue
                b = time.time()
                time_evaluation[iteration].append(b - a)

                next_generation_prompts = []
                number_of_generations_by_mutations = 2
                number_of_generations_by_crossover = 2
                straight_of_generations_by_mutations = 1
                ## mutation
                selected_candidates_for_mutations = choose_candidates(candidates.copy(), number_of_generations_by_mutations)
                for a_candidate in selected_candidates_for_mutations:
                    llama_prompts_final = mutate_prompts_api(a_candidate, mutation_llm)
                    # time.sleep(3)
                    next_generation_prompts.append(llama_prompts_final)
                ##crossover
                for j in range(number_of_generations_by_crossover):
                    llama_prompts_final = crossover_prompts_api(candidates.copy(), mutation_llm)
                    # time.sleep(3)
                    next_generation_prompts.append(llama_prompts_final)
                ## straight select
                next_generation_prompts.extend(choose_candidates(candidates.copy(), straight_of_generations_by_mutations))
                # print(f'second nexxxxxxxxxxxxxxxxxxxxxxxxxx for {idx}')
                # print(next_generation_prompts[1])
                base_prompts_re[idx] = next_generation_prompts

            d = time.time()
            time_next_make_generation[iteration].append(d - b)
            time_total_per_instance[iteration].append(d - c)
        chosen_prompts = [rr[0] for rr in base_prompts_re[0:number_of_tests]]  ##here
        ## evaluation
        if run_evaluation_each_generation:
            run_final_evaluation(chosen_prompts, codeLLama_model, codeLLama_tokenizer, evaluations, final_test_cases,
                                 human_eval, iteration, magic_coder, model_to_test, number_of_tests, passed_codes,
                                 time_test)
        ## evaluations
        print_time_measures(evaluations, number_of_supposed_passed_codes, start, time_evaluation,
                            time_next_make_generation,
                            time_test, time_total_per_instance)

    print_time_measures(evaluations, number_of_supposed_passed_codes, start, time_evaluation, time_next_make_generation,
                        time_test, time_total_per_instance)


def stop_criteria_met(number_of_supposed_passed_codes, dataset_length, iteration):
    if iteration >= 3:
        return True
    else:
        return False


def run_genetic_algorithm_gensim(base_prompts_re, codeLLama_tokenizer, codeLLama_model, magic_coder, final_test_cases, generated_testcases, human_eval, number_of_tests=164, model_to_test=0, gpt_client=None, population_size=5):

    all_generated_promts = []
    # all_generated_promts = []
    evaluations = []
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import warnings
    warnings.filterwarnings("ignore")
    iteration = 0
    run_evaluation_each_generation = True
    ## time management
    time_total_per_instance = []
    time_evaluation = []
    time_test = []
    time_next_make_generation = []
    number_of_supposed_passed_codes = []
    # if model_to_test == 1:
    #     base_prompts_re = base_prompts_re_codemagic.copy()

    passed_codes = [False for i in range(number_of_tests)]
    start = time.time()
    while(not stop_criteria_met(evaluations)):
        time_total_per_instance.append([])
        time_evaluation.append([])
        time_next_make_generation.append([])

        all_generated_promts.append(base_prompts_re.copy())
        number_of_supposed_passed_codes.append(0)
        for idx, a_prompt_set in tqdm(enumerate(base_prompts_re[0:number_of_tests])):  ##here
            print(idx)
            c = time.time()
            passed = False
            if len(a_prompt_set) == 1:
                time_total_per_instance[iteration].append(0)
                time_evaluation[iteration].append(0)
                time_next_make_generation[iteration].append(0)
                passed = True
                number_of_supposed_passed_codes[iteration] += 1
                continue
            else:
                candidates = []
                a = time.time()
                for single_prompt in a_prompt_set:
                    passed = False
                    passat1, filling = evaluate_prompt_on_generated_prompts(
                        generated_test_cases=generated_testcases[idx][0:4],
                        prompt=single_prompt, model_to_test=model_to_test,
                        prompt_index=idx,
                        codeLLama_tokenizer=codeLLama_tokenizer,
                        codeLLama_model=codeLLama_model,
                        magic_coder=magic_coder,
                        human_eval=human_eval,
                        gpt_client=gpt_client)

                    candidates.append([single_prompt, passat1])
                    if passat1 == 1:
                        base_prompts_re[idx] = [single_prompt]
                        passed_codes[idx] = filling
                        print(
                            f'PAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAASED for idx {idx}')
                        passed = True
                        break
                if passed:
                    b = time.time()
                    time_evaluation[iteration].append(b-a)
                    time_total_per_instance[iteration].append(b-a)
                    time_next_make_generation[iteration].append(0)
                    continue
                b = time.time()
                time_evaluation[iteration].append(b - a)

                next_generation_prompts = []
                if population_size == 5:
                    number_of_generations_by_mutations = 4
                    straight_of_generations_by_mutations = 1
                elif population_size == 10:
                    number_of_generations_by_mutations = 8
                    straight_of_generations_by_mutations = 2
                ## straight select
                next_generation_prompts.extend(choose_candidates(candidates, straight_of_generations_by_mutations))

                ## mutation
                selected_candidates_for_mutations = choose_candidates(candidates, number_of_generations_by_mutations)
                for a_candidate in selected_candidates_for_mutations:
                    splits = a_candidate.split(special_token)
                    if len(splits) != 5:
                        alternate_sentences = mutate_sentence(splits[1], num_versions=1,
                                                              similarity_threshold=0.5)
                        final_sentence = splits[0] + special_token + alternate_sentences[0] + special_token + splits[2]
                    else:
                        alternate_sentences1 = mutate_sentence(splits[1], num_versions=1,
                                                               similarity_threshold=0.5)
                        alternate_sentences2 = mutate_sentence(splits[3], num_versions=1,
                                                               similarity_threshold=0.5)
                        final_sentence = splits[0] + special_token + alternate_sentences1[0] + special_token + splits[
                            2] + special_token + alternate_sentences2[0] + special_token + splits[4]
                    next_generation_prompts.append(final_sentence)
                # print(f'nexxxxxxxxxxxxxxxxxxxxxxxxxx for {idx}')
                # print(next_generation_prompts)
                base_prompts_re[idx] = next_generation_prompts

            d = time.time()
            time_next_make_generation[iteration].append(d - b)
            time_total_per_instance[iteration].append(d - c)

        chosen_prompts = [rr[0] for rr in base_prompts_re[0:number_of_tests]]  ##here
        ## evaluation
        if run_evaluation_each_generation:
            run_final_evaluation(chosen_prompts, codeLLama_model, codeLLama_tokenizer, evaluations, final_test_cases,
                                 human_eval, iteration, magic_coder, model_to_test, number_of_tests, passed_codes,
                                 time_test, gpt_client)
        ## evaluations
        iteration += 1
    print_time_measures(evaluations, number_of_supposed_passed_codes, start, time_evaluation, time_next_make_generation,
                        time_test, time_total_per_instance)
    print('successful prompts **********************************************************')
    print(base_prompts_re)
    print('successful codes ****************************************************************')
    print(passed_codes)


def select_final_prompts(base_prompts_re, dataset):
    chosen_prompts = []
    for index,prompt_set in enumerate(base_prompts_re):
        if len(prompt_set) == 1:
            chosen_prompts.append(prompt_set[0])
        else:
            chosen_prompts.append(dataset[index])
    return chosen_prompts


def save_results(dataset_choice, dataset, final_code, errors_index, chosen_prompts, final_test_cases, seed, model_to_test, experiment_to_run):
    file_dir = f'output/{experiment_to_run}.jsonl'
    out_dict = []
    bigloader = BigCodeLoader(hard=1)
    bigcode_dis = bigloader.get_ids()
    for index, item in enumerate(dataset):
        out_dict.append({
            'prompt': item,
            'solution': final_code[index][0],
            'test_cases': final_test_cases[index],
            'name': index,
            'task_id': index,
            'is_passed': 'False' if index in errors_index else 'True'
        })
        if dataset_choice == 3:
            out_dict[-1]['name'] = bigcode_dis[index]
            out_dict[-1]['task_id'] = bigcode_dis[index]
    with open(file_dir, "w") as file:
        for item in out_dict:
            file.write(json.dumps(item) + "\n")


def run_genetic_algorithm_gensim_(codeLLama_tokenizer, codeLLama_model, magic_coder, final_test_cases, generated_testcases, dataset,experiment_to_run, number_of_tests=164, model_to_test=0, gpt_client=None, population_size=5, dataset_choice=1, seed=137, mutation_tool=1, model_name='o3-mini-2025-01-31'):

    random.seed(seed)
    all_generated_promts = []
    # all_generated_promts = []
    evaluations = []
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import warnings
    warnings.filterwarnings("ignore")
    iteration = 0
    run_evaluation_each_generation = False
    ## time management
    time_total_per_instance = []
    time_evaluation = []
    time_test = []
    time_next_make_generation = []
    number_of_supposed_passed_codes = []
    # if model_to_test == 1:
    #     base_prompts_re = base_prompts_re_codemagic.copy()
    # results_list = [{'task_id': i} for i in dataset]
    passed_codes = [False for i in range(number_of_tests)]
    start = time.time()

    ## pre evaluation
    base_prompts_re = []
    from results.gpt_humaneval_code_completion import gpt_generated_codes
    time_total_per_instance.append([])
    time_evaluation.append([])
    time_next_make_generation.append([])
    pass_threshold = 1
    total_usage = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
    }
    all_codes = [[]]
    for i in range(len(dataset)):
        all_codes[iteration].append([])
    for idx, prompt in enumerate(dataset):
        time_one = time.time()
        # print(f'here1 {idx}')
        passat1, filling, usage = evaluate_prompt_on_generated_prompts(
            generated_test_cases=generated_testcases[idx],
            prompt=prompt, model_to_test=model_to_test,
            prompt_index=idx,
            codeLLama_tokenizer=codeLLama_tokenizer,
            codeLLama_model=codeLLama_model,
            magic_coder=magic_coder,
            human_eval=dataset,
            gpt_client=gpt_client,
            model_name=model_name,
            dataset_choice=dataset_choice
        )

        total_usage['prompt_tokens'] += usage['prompt_tokens']
        total_usage['completion_tokens'] += usage['completion_tokens']
        total_usage['total_tokens'] += usage['total_tokens']
        time_evaluation[iteration].append(round(time.time() - time_one))
        # print(idx)
        # print(passat1)
        # print(filling)
        all_codes[iteration][idx].append((prompt, filling, passat1))
        if passat1 >= 1:
            base_prompts_re.append([prompt])
            passed_codes[idx] = filling
            print(
                f'PAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAASED for idx {idx}')
            time_next_make_generation[iteration].append(0)
        else:
            time_a = time.time()
            first_generation, usage2 = generate_first_population_for_instance(prompt=prompt,population_size=population_size,client=gpt_client, human_eval=dataset,use_stored_prompts=False, idx=idx, dataset_choice=dataset_choice, generated_testcases=generated_testcases[idx], model_name=model_name, model_to_test=model_to_test)

            total_usage['prompt_tokens'] += usage2['prompt_tokens']
            total_usage['completion_tokens'] += usage2['completion_tokens']
            total_usage['total_tokens'] += usage2['total_tokens']
            base_prompts_re.append(first_generation)
            time_next_make_generation[iteration].append(time.time() - time_a)
        time_two = time.time()
        time_total_per_instance[iteration].append(round(time_two - time_one))
    chosen_prompts = select_final_prompts(base_prompts_re, dataset)
    # chosen_prompts = [rr[0] for rr in base_prompts_re]
    if run_evaluation_each_generation:
        run_final_evaluation(chosen_prompts, codeLLama_model, codeLLama_tokenizer, evaluations, final_test_cases,
                             dataset, iteration, magic_coder, model_to_test, number_of_tests, passed_codes,
                             time_test, model_name,dataset_choice, gpt_client)
    print('initial evaluation and making first generation time in seconds: ', round(time.time() - start))
    # pre evaluation
    iteration += 1
    while not stop_criteria_met(number_of_supposed_passed_codes, len(dataset), iteration):
        all_codes.append([])
        for i in range(len(dataset)):
            all_codes[iteration].append([])

        time_total_per_instance.append([])
        time_evaluation.append([])
        time_next_make_generation.append([])

        all_generated_promts.append(base_prompts_re.copy())
        number_of_supposed_passed_codes.append(0)
        for idx, a_prompt_set in tqdm(enumerate(base_prompts_re[0:number_of_tests])):
            print(idx)
            c = time.time()
            passed = False
            if len(a_prompt_set) == 1:
                time_total_per_instance[iteration].append(0)
                time_evaluation[iteration].append(0)
                time_next_make_generation[iteration].append(0)
                passed = True
                number_of_supposed_passed_codes[iteration-1] += 1
                continue
            else:
                candidates = []
                a = time.time()
                for single_prompt in a_prompt_set:
                    passed = False
                    passat1, filling, usage = evaluate_prompt_on_generated_prompts(
                        generated_test_cases=generated_testcases[idx],
                        prompt=single_prompt, model_to_test=model_to_test,
                        prompt_index=idx,
                        codeLLama_tokenizer=codeLLama_tokenizer,
                        codeLLama_model=codeLLama_model,
                        magic_coder=magic_coder,
                        human_eval=dataset,
                        gpt_client=gpt_client,
                        model_name=model_name,
                        dataset_choice=dataset_choice)
                    total_usage['prompt_tokens'] += usage['prompt_tokens']
                    total_usage['completion_tokens'] += usage['completion_tokens']
                    total_usage['total_tokens'] += usage['total_tokens']
                    all_codes[iteration][idx].append((single_prompt, filling, passat1))
                    candidates.append([single_prompt, passat1, filling])
                    if passat1 == 1:
                        base_prompts_re[idx] = [single_prompt]
                        passed_codes[idx] = filling
                        print(
                            f'PAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAASED for idx {idx}')
                        passed = True
                        break
                if passed:
                    b = time.time()
                    time_evaluation[iteration].append(b-a)
                    time_total_per_instance[iteration].append(b-a)
                    time_next_make_generation[iteration].append(0)
                    continue

                best_candidate = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
                if best_candidate[1] >= pass_threshold:
                    base_prompts_re[idx] = [best_candidate[0]]
                    passed_codes[idx] = best_candidate[2]
                    print(
                        f'PAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAASED for idx {idx}')
                    passed = True
                    if passed:
                        b = time.time()
                        time_evaluation[iteration].append(b - a)
                        time_total_per_instance[iteration].append(b - a)
                        time_next_make_generation[iteration].append(0)
                        continue

                b = time.time()
                time_evaluation[iteration].append(b - a)

                next_generation_prompts = []
                if population_size == 5:
                    number_of_generations_by_mutations = 4
                    straight_of_generations_by_mutations = 1
                elif population_size == 10:
                    number_of_generations_by_mutations = 9
                    straight_of_generations_by_mutations = 1
                elif population_size == 3:
                    number_of_generations_by_mutations = 2
                    straight_of_generations_by_mutations = 1
                elif population_size == 8:
                    number_of_generations_by_mutations = 7
                    straight_of_generations_by_mutations = 1
                ## straight select
                # print(candidates)
                # print('*'*100)
                aaa = choose_candidates(candidates, straight_of_generations_by_mutations)
                next_generation_prompts.extend([a[0] for a in aaa])

                ## mutation
                selected_candidates_for_mutations = choose_candidates(candidates, number_of_generations_by_mutations)
                for a_candidate in selected_candidates_for_mutations:
                    if mutation_tool == 1:
                        final_sentence = mutate_prompt(a_candidate)
                        # final_sentence = augment_promt(a_candidate)
                    else:
                        final_sentence, usage = mutate_prompt_gpt(a_candidate, gpt_client, model_name)
                        total_usage['prompt_tokens'] += usage['prompt_tokens']
                        total_usage['completion_tokens'] += usage['completion_tokens']
                        total_usage['total_tokens'] += usage['total_tokens']
                        # final_sentence = mutate_prompt_gpt_v2(a_candidate, gpt_client, model_name)
                    next_generation_prompts.append(final_sentence)
                base_prompts_re[idx] = next_generation_prompts

            d = time.time()
            time_next_make_generation[iteration].append(d - b)
            time_total_per_instance[iteration].append(d - c)

        chosen_prompts = select_final_prompts(base_prompts_re, dataset)  ##here
        ## evaluation
        if run_evaluation_each_generation:
            final_code, errors_index = run_final_evaluation(chosen_prompts, codeLLama_model, codeLLama_tokenizer, evaluations, final_test_cases,
                                 dataset, iteration, magic_coder, model_to_test, number_of_tests, passed_codes,
                                 time_test,model_name,dataset_choice, gpt_client)
        iteration += 1
    if not run_evaluation_each_generation:
        final_code, errors_index = run_final_evaluation(chosen_prompts, codeLLama_model, codeLLama_tokenizer, evaluations, final_test_cases,
                             dataset, iteration, magic_coder, model_to_test, number_of_tests, passed_codes,
                             time_test, model_name,dataset_choice,gpt_client)
    # print(passed_codes)
    # print('Final prompts:-----------------------------------')
    # print(chosen_prompts)
    # print('Final codes:---------------------------------------')
    # print(final_code)
    # print('All prompts:---------------------------------------')
    # print(all_codes)
    print_time_measures(evaluations, number_of_supposed_passed_codes, start, time_evaluation, time_next_make_generation,
                        time_test, time_total_per_instance, total_usage)
    save_results(dataset_choice, dataset, final_code, errors_index, chosen_prompts, final_test_cases, seed, model_to_test, experiment_to_run)
    # return evaluations[-1][0]['pass@1']
    # print('successful prompts **********************************************************')
    # print(base_prompts_re)
    # print('successful codes ****************************************************************')
    # print(passed_codes)