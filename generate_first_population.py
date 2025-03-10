import openai
import time
from chat_gpt_prompts_distilled import get_gpt_prompts_distilled, refactor_prompt
from gensimutils import mutate_prompt
from dotenv import load_dotenv
import os
import anthropic
import requests
import json
load_dotenv()
key = os.getenv('openai_key')

def get_first_population(gpt_prompts, human_eval, population_size, idx):
    a = gpt_prompts[0:population_size]
    b = [human_eval[idx]]
    b.extend(a)
    return b


def generate_first_population_for_instance(prompt, population_size, idx, client, human_eval, dataset_choice,generated_testcases, model_name,model_to_test,use_stored_prompts=True):
    usage ={
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
    prompt, usage = generate_first_population(prompt, population_size, client, dataset_choice, generated_testcases, model_name,model_to_test)
    first_generation = refactor_prompt(get_first_population(prompt, human_eval, population_size, idx))
    return first_generation, usage


def get_completion(client, prompt, population_size,model_name, model_to_test,dataset_choice=1):
    if dataset_choice == 1:
        prompt1 = """Please rewrite the function description based on these instructions:
            1- Add input and output types of the function to the description.
            2- Elaborate the description so that it is understandable for large language models.
            3- Keep the original testcases and add 3 test cases to the description to cover the edge cases. Do not separate the generated testcases and the original ones.
            Keep the structure of the function and add the description as a comment in the function. Use at most 600 words. Do not implement the code\n"""
        max_tokens = 1000
    else:
        prompt1 = """Please rewrite the function description based on these instructions:
            1- Add input and output types of the function to the description.
            2- Elaborate the description so that it is understandable for large language models.
            Keep the structure of the function and add the description as a comment in the function. Use at most 800 words. Do not implement the code. Put the rewritten function description (and it's signature) in between ```python and ``` tags.\n"""
        max_tokens = 1000
    if model_to_test == 4:
        client = anthropic.Anthropic(api_key=os.getenv("anthropic_key"))
        response = client.messages.create(
            model=model_name,
            max_tokens=1500,
            temperature=0.8,
            # system="You are an expert python developer",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt1  + '\n\n'+ prompt,
                        }
                    ]
                }
            ]
        )
    elif model_to_test == 5:
        fire_work_key = os.getenv('fireworks_key')
        url = "https://api.fireworks.ai/inference/v1/chat/completions"
        payload = {
            "model": f"accounts/fireworks/models/{model_name}",
            "max_tokens": 8000,
            # "top_p": 1,
            # "top_k": 40,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "temperature": 0.6,
            "n": population_size,
            "messages": [
                {
                    "role": "user",
                    "content":  prompt1  + '\n\n'+ prompt
                }
            ]
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {fire_work_key}"
        }
        response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    else:
        response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt1+prompt}],
                # temperature=0.7,
                # max_completion_tokens=max_tokens,
                n=population_size
            )
    return response


def generate_first_population(prompt, population_size, client, dataset_choice, generated_testcases, model_name, model_to_test):
    """
    :param dataset_choice: 1:humnaeval, 2:mbpp
    :param generated_testcases:
    :param prompt:
    :param population_size:
    :param client:
    :return:
    """
    if model_to_test == 4:
        all_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        first_generation = []
        for i in range(population_size):
            response = get_completion(client, prompt, population_size, model_name,model_to_test, dataset_choice)
            text = response.content[0].text
            all_usage['prompt_tokens'] += response.usage.input_tokens
            all_usage['completion_tokens'] += response.usage.output_tokens
            all_usage['total_tokens'] += response.usage.input_tokens + response.usage.output_tokens
            try:
                # print(a_prompt.message.content)
                try:
                    res = text.split("```")[1].replace('python\n', '')
                except IndexError:
                    res = text
                prompt_splits = res.split('"""')
                test_text = ""
                for a_test in generated_testcases[:3]:
                    test_text += "- " + a_test + " \n"
                prompt_splits[1] = prompt_splits[1] + "\nTestcases:\n" + test_text
                new_prompt = prompt_splits
                first_generation.append('"""'.join(new_prompt))
            except IndexError:
                # print(f"index error")
                new_prompt = prompt
                first_generation.append(new_prompt)

        return first_generation, all_usage

    elif model_to_test == 5:
        response = get_completion(client, prompt, population_size, model_name,model_to_test, dataset_choice)
        response = response.json()

        usage = response['usage']
        first_generation = []
        for a_prompt in response['choices']:
            try:
                # print(a_prompt.message.content)
                try:
                    message = a_prompt['message']['content'].split('</think>')[1]
                except IndexError:
                    message = a_prompt['message']['content']
                try:
                    res = message.split("```")[1].replace('python\n', '')
                except IndexError:
                    res = message
                prompt_splits = res.split('"""')
                test_text = ""
                for a_test in generated_testcases[:3]:
                    test_text += "- " + a_test + " \n"
                prompt_splits[1] = prompt_splits[1] + "\nTestcases:\n" + test_text
                new_prompt = prompt_splits
                first_generation.append('"""'.join(new_prompt))
            except IndexError:
                # print(f"index error")
                new_prompt = prompt
                first_generation.append(new_prompt)
                # return first_generation, usage
        return first_generation, usage

    else:
        number_of_tries = 5
        counter = 0
        usage = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0,
        }
        while True:
            try:
                if counter == number_of_tries:
                    print(f'generating first population failed for prompt: {prompt}')
                    return [prompt] * population_size, usage
                counter += 1
                response = get_completion(client, prompt, population_size,model_name, dataset_choice)
                break
            except openai.InternalServerError:
                print('Internal Server Error, waiting 10 seconds...')
                time.sleep(10)
            except openai.BadRequestError as e:
                print(e)
                print('Bad Request Error')
                return [prompt] * population_size, usage
        first_generation = []
        usage = response.usage
        usage = {
            'completion_tokens': usage.completion_tokens,
            'prompt_tokens': usage.prompt_tokens,
            'total_tokens': usage.total_tokens,
        }
        for a_prompt in response.choices:
            try:
                # print(a_prompt.message.content)
                try:
                    res = a_prompt.message.content.split("```")[1].replace('python\n', '')
                except IndexError:
                    res = a_prompt.message.content
                prompt_splits = res.split('"""')
                test_text = ""
                for a_test in generated_testcases[:3]:
                    test_text += "- " + a_test + " \n"
                prompt_splits[1] = prompt_splits[1] + "\nTestcases:\n" + test_text
                new_prompt = prompt_splits
                first_generation.append('"""'.join(new_prompt))
            except IndexError:
                # print(f"index error")
                new_prompt = prompt
                first_generation.append(new_prompt)
                # return first_generation, usage
        return first_generation, usage
