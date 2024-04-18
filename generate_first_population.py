from chat_gpt_prompts_distilled import get_gpt_prompts_distilled, refactor_prompt
from chat_gpt_prompts_distilled_ten_generation import get_gpt_prompts_distilled_ten


def get_first_population(gpt_prompts, human_eval, population_size, idx):
    a = gpt_prompts[0:population_size]
    b = [human_eval[idx]]
    b.extend(a)
    return b


def generate_first_population_for_instance(prompt, population_size, idx, client, human_eval, use_stored_prompts=True):
    if use_stored_prompts:
        if population_size == 5:
            prompt = get_gpt_prompts_distilled()[idx]
        else:
            prompt = get_gpt_prompts_distilled_ten()[idx]
    else:
        prompt = generate_first_population(prompt, population_size, client)
    first_generation = refactor_prompt(get_first_population(prompt, human_eval, population_size, idx))
    return first_generation

def generate_first_population(prompt, population_size, client):
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[{"role": "user", "content":
            """Please rewrite the function description based on these instructions:
            1- Add input and output types of the function to the description.
            2- Elaborate the description so that it is understandable for large language models.
            3- Explore if there are any exceptional or edge cases and elaborate on them. Ignore input validation and focus on cases which clarifies the problem.
            4- Keep the original testcases and add enough test cases to the description to cover the edge cases. Do not separate the generated testcases and the original ones.
            Keep the structure of the function and add the description as a comment in the function. Use at most 800 words,\n""" +
            prompt}],
        temperature=0.5,
        max_tokens=1000,
        n=population_size
    )
    first_generation = []
    for a_prompt in response.choinces:
        first_generation.append(a_prompt.message.content.split("```")[1].replace('python\n', ''))

    return first_generation