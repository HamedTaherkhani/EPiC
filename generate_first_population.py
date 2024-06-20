from chat_gpt_prompts_distilled import get_gpt_prompts_distilled, refactor_prompt
from chat_gpt_prompts_distilled_ten_generation import get_gpt_prompts_distilled_ten
from gensimutils import mutate_prompt
from dotenv import load_dotenv
import os
load_dotenv()
key = os.getenv('openai_key')

openai_model = os.getenv("openai_model")

def get_first_population(gpt_prompts, human_eval, population_size, idx):
    a = gpt_prompts[0:population_size]
    b = [human_eval[idx]]
    b.extend(a)
    return b


def generate_first_population_for_instance(prompt, population_size, idx, client, human_eval, dataset_choice,generated_testcases,use_stored_prompts=True):
    if use_stored_prompts:
        if population_size == 5:
            prompt = get_gpt_prompts_distilled()[idx]
        else:
            prompt = get_gpt_prompts_distilled_ten()[idx]
    else:
        prompt = generate_first_population(prompt, population_size, client, dataset_choice, generated_testcases)
    first_generation = refactor_prompt(get_first_population(prompt, human_eval, population_size, idx))
    return first_generation


def generate_first_population_for_instance_v2(prompt, population_size, client):
    first_generation = []
    first_prompt = generate_first_population_v2(prompt, client)
    first_generation.append(first_prompt)
    prompt_re = refactor_prompt([first_prompt])[0]

    for i in range(population_size - 1):
        first_generation.append(mutate_prompt(prompt_re))
    return refactor_prompt(first_generation)


def get_completion(client, prompt, population_size, dataset_choice=1):
    if dataset_choice == 1:
        prompt1 = """Please rewrite the function description based on these instructions:
            1- Add input and output types of the function to the description.
            2- Elaborate the description so that it is understandable for large language models.
            3- Keep the original testcases and add enough test cases to the description to cover the edge cases. Do not separate the generated testcases and the original ones.
            Keep the structure of the function and add the description as a comment in the function. Use at most 500 words. Do not implement the code\n"""
        max_tokens = 600
    else:
        prompt1 = """Please rewrite the function description based on these instructions:
            1- Add input and output types of the function to the description.
            2- Elaborate the description so that it is understandable for large language models.
            Keep the structure of the function and add the description as a comment in the function. Use at most 250 words. Do not implement the code\n"""
        max_tokens = 400
    response = client.chat.completions.create(
        model=openai_model,
        messages=[{"role": "user", "content": prompt1+prompt}],
        temperature=0.9,
        max_tokens=max_tokens,
        n=population_size
    )
    return response


def generate_first_population(prompt, population_size, client, dataset_choice, generated_testcases):
    """
    :param dataset_choice: 1:humnaeval, 2:mbpp
    :param generated_testcases:
    :param prompt:
    :param population_size:
    :param client:
    :return:
    """
    response = get_completion(client, prompt, population_size, dataset_choice)
    first_generation = []
    for a_prompt in response.choices:
        try:
            prompt = a_prompt.message.content.split("```")[1].replace('python\n', '')
        except IndexError:
            print("index error")
            print(a_prompt.message.content)
            prompt = a_prompt.message.content
        prompt_splits = prompt.split('"""')
        test_text = ""
        for a_test in generated_testcases[:3]:
            test_text += "- " + a_test + " \n"
        prompt_splits[1] = prompt_splits[1] + "\nTestcases:\n" + test_text
        first_generation.append('"""'.join(prompt_splits))

    return first_generation


def generate_first_population_v2(prompt, client):
    response = get_completion(client, prompt, 1).choices[0].message.content.split("```")[1].replace('python\n', '')
    return response
