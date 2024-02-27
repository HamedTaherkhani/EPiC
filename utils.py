import requests
import re
from tqdm import tqdm
from evaluate import load
import time
import random
import numpy as np
from multiprocessing import Pool
from itertools import repeat
from itertools import product
from gensimutils import provide_alternate_sentence
max_response_length = 2500
def f(a_test, candidates):
    print(a_test)
    print(candidates)
    pass_at_k, results = code_eval_metric.compute(references=[a_test], predictions=candidates, k=[1])
    return pass_at_k


special_token = "#SPECIAL_TOKEN"
code_eval_metric = load("code_eval")
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def choose_candidates(prompts_set, number=1):
    chosen_prompts = []
    try:
        for i in range(number):
            # temp = random.choices(prompts_set, weights=((ss[2]+ss[1]) for ss in prompts_set), k=1)[0]
            temp = random.choices(prompts_set, weights=(ss[1] for ss in prompts_set), k=1)[
                0]  ## just considering pass@k
            prompts_set.remove(temp)
            chosen_prompts.append(temp)
    except ValueError:
        chosen_prompts = random.choices(prompts_set, k=number)
    return [a[0] for a in chosen_prompts]

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
    if len(res) > max_response_length + 1:
        res = res[0:max_response_length]
    exp = [m.start() for m in re.finditer(special_token, a_candidate)]
    if len(exp) == 0:
        exp = [m.start() for m in re.finditer("'''", a_candidate)]
    if len(exp) == 1:
        exp.append(-1)
    elif len(exp) == 0:
        exp = [0, -1]
    final_prompt = a_candidate.replace(a_candidate[exp[0] + 17: exp[1]], res)
    return final_prompt


# def process_prompt(res, a_candidate):

headers = {
    'Content-Type': 'text/plain'
}


def mutate_prompts_api(a_candidate):
    query2 = "Here is a python function and it's description. Please Refine and elaborate the description by enhancing it's clarity and comprehension for sophisticated language models. Please put the refined description between #Explanation and #End. \\n"
    url = "https://www.llama2.ai/api"
    prompt_changed = a_candidate.replace('\n', '\\n').replace("\"", '\\"')
    payload = "{\"prompt\":\"[INST]Hello [/INST]\\n\",\"model\":\"meta/llama-2-70b-chat\",\"systemPrompt\":\"You are a helpful assistant.\",\"temperature\":0.75,\"topP\":0.9,\"maxTokens\":300,\"image\":null,\"audio\":null}"
    payload = payload.replace("Hello", query2 + prompt_changed)

    response = requests.request("POST", url, headers=headers, data=payload).text
    return process_prompt2(response, a_candidate)


def crossover_prompts_api(cands):
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
    payload = "{\"prompt\":\"[INST]Hello [/INST]\\n\",\"model\":\"meta/llama-2-70b-chat\",\"systemPrompt\":\"You are a helpful assistant.\",\"temperature\":0.75,\"topP\":0.9,\"maxTokens\":300,\"image\":null,\"audio\":null}"
    payload = payload.replace("Hello", prompt_changed)
    response = requests.request("POST", url, headers=headers, data=payload).text
    return process_api_prompt(response, two_candidates[0])


def mutate_and_return_candidate(cands, generate_text):
    a_candidate = choose_candidates(cands, 1)[0]
    # print(a_candidate)
    query2 = f"Change the explanation of the function to ensure clarity and comprehension for sophisticated language models. Please put the explanation after # Explanation: and before # End .\n" + a_candidate
    # print(query2)
    res = generate_text(query2)[0]['generated_text']
    llama_prompts_final = process_prompt(res, a_candidate)
    return llama_prompts_final


def crossover_and_return_candidate(cands, generate_text):
    two_candidates = choose_candidates(cands, 2)
    PROMPT = 'Merge the first explanation and the second explanation to have a new explanation for the function. Please put the new explanation after # Explanation: and before # End .\n' + 'first explanation:\n' + \
             two_candidates[0] + 'second explanation:\n' + two_candidates[1]
    res = generate_text(PROMPT)[0]['generated_text']
    llama_prompts_final = process_prompt(res, two_candidates[0])
    return llama_prompts_final


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


def evaluate_prompt_on_generated_prompts(test_cases, prompt, codeLLama_tokenizer, codeLLama_model, magic_coder, human_eval,
                                         model_to_test=0, prompt_index=None):
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
            return 0
    elif model_to_test == 1:
        filling = \
        magic_coder(prompt.replace('#SPECIAL_TOKEN', ''), max_length=512, num_return_sequences=1, do_sample=False)[0][
            'generated_text']
        filling = process_a_code_magic_coder(filling, prompt_index,human_eval)
    ##
    candidate = [filling]
    candidates = [candidate]

    pass_total = 0
    for a_test in test_cases:
        pass_at_k, results = code_eval_metric.compute(references=[a_test], predictions=candidates, k=[1])
        pass_total += pass_at_k['pass@1']
    return pass_total / len(test_cases)

    # with Pool() as p:
    #     results = p.starmap(f, zip(test_cases, repeat(candidates)))
    # print(results)
    # return sum(results) / len(results)


def run_final_evaluation(chosen_prompts, codeLLama_model, codeLLama_tokenizer, evaluations, final_test_cases,
                         human_eval, iteration, magic_coder, model_to_test, number_of_tests, passed_codes, time_test):
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
                if not passed_codes[index]:
                    filling = magic_coder(a_token.replace('#SPECIAL_TOKEN', ''), max_length=800, num_return_sequences=1,
                                          temperature=0.0)[0]['generated_text']
                    fillings.append(filling)
                else:
                    fillings.append(passed_codes[index])
            fillings = process_the_code_magic_coder(fillings, human_eval)
            fillings = [[fil] for fil in fillings]
        errorrrs = []
        pass_at_k, results = code_eval_metric.compute(references=final_test_cases[0:number_of_tests],
                                                      predictions=fillings, k=[1])  ##here
        for key, value in results.items():
            if value[0][1]['passed']:
                passed_codes[value[0][1]['task_id']] = fillings[value[0][1]['task_id']][0]
        evaluations.append((pass_at_k, results))
        print(pass_at_k)
        time_test.append(time.time() - e)
        # for key,item in results[1].items():
        #     if item[0][1]['passed']:
        #         passed_fillings[item[0][1]['task_id']] = fillings[item[0][1]['task_id']][0]


def print_time_measures(evaluations, number_of_supposed_passed_codes, start, time_evaluation, time_next_make_generation,
                        time_test, time_total_per_instance):
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
    print('time_total_per_instance for every loop:')
    print(np.sum(time_total_per_instance, axis=1) + time_test)
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


def run_genetic_algorithm(base_prompts_re, codeLLama_tokenizer, codeLLama_model, magic_coder, final_test_cases, generated_testcases, human_eval, number_of_tests=164, model_to_test=0):
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
                    passat10 = evaluate_prompt_on_generated_prompts(test_cases=generated_testcases[idx][0:4],
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
                    llama_prompts_final = mutate_prompts_api(a_candidate)
                    # time.sleep(3)
                    next_generation_prompts.append(llama_prompts_final)
                ##crossover
                for j in range(number_of_generations_by_crossover):
                    llama_prompts_final = crossover_prompts_api(candidates.copy())
                    # time.sleep(3)
                    next_generation_prompts.append(llama_prompts_final)
                ## straight select
                next_generation_prompts.extend(choose_candidates(candidates.copy(), straight_of_generations_by_mutations))
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
                                 time_test)
        ## evaluations
    print_time_measures(evaluations, number_of_supposed_passed_codes, start, time_evaluation, time_next_make_generation,
                        time_test, time_total_per_instance)


def run_genetic_algorithm_gensim(base_prompts_re, codeLLama_tokenizer, codeLLama_model, magic_coder, final_test_cases, generated_testcases, human_eval, number_of_tests=164, model_to_test=0):

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
                    passat10 = evaluate_prompt_on_generated_prompts(test_cases=generated_testcases[idx][0:3],
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
                number_of_generations_by_mutations = 4
                straight_of_generations_by_mutations = 1
                ## straight select
                next_generation_prompts.extend(choose_candidates(candidates, straight_of_generations_by_mutations))

                ## mutation
                selected_candidates_for_mutations = choose_candidates(candidates, number_of_generations_by_mutations)
                for a_candidate in selected_candidates_for_mutations:
                    splits = a_candidate.split(special_token)
                    if len(splits) != 5:
                        alternate_sentences = provide_alternate_sentence(splits[1], num_versions=1,
                                                                         similarity_threshold=0.5)
                        final_sentence = splits[0] + special_token + alternate_sentences[0] + special_token + splits[2]
                    else:
                        alternate_sentences1 = provide_alternate_sentence(splits[1], num_versions=1,
                                                                          similarity_threshold=0.5)
                        alternate_sentences2 = provide_alternate_sentence(splits[3], num_versions=1,
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
                                 time_test)
        ## evaluations
    print_time_measures(evaluations, number_of_supposed_passed_codes, start, time_evaluation, time_next_make_generation,
                        time_test, time_total_per_instance)