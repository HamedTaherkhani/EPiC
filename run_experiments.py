import os
import json
# os.environ['TRANSFORMERS_CACHE'] = '/home/hamedth/projects/def-hemmati-ac/hamedth/hugging_face'
from magiccoder_experiments import MagicCoderRunner
from gpt_experiments import GPTRunner
from llama_experiments import LlamaExperiments
from antropic_experiments import AntropicRunner
from fireworks_experiments import FireworksExperiments
from dotenv import load_dotenv
import sys
import random


load_dotenv()
from codellama_experiments import CodellamaExperiments
experiments = {
    1: 'genetic-magiccoder-llama2-70b',
    2: 'genetic-codellama-llama2-70b',
    3: 'genetic-magiccoder-gensim',
    4: 'genetic-codellama-gensim',
    5: 'genetic-magicoder-llama2-7b',
    6: 'genetic-o3mini-humaneval',
    7: 'genetic-o3mini-eight_population',
    8: 'genetic-gpt4-gensim-10-times',
    9: 'genetic-o3mini-mbpp',
    10: 'genetic-gpt4-gpt4',
    11: 'genetic-gpt4-gensim-mbpp-10-times',
    12: 'genetic-llama3-8b-humaneval',
    13: 'genetic-deepseek-humaneval',
    14: 'genetic-o3mini-bigcodehard',
    15: 'genetic-o3mini-bigcodehard-llm',
    16: 'genetic-o3mini-bigcodehard-simple-llm',
    17: 'genetic-sonnet3.7-bigcodehard',
    18: 'genetic-deepseek-bigcodehard',
    19: 'genetic-deepseek-humaneval',
    20: 'genetic-sonnet3.7-humaneval',
    21: 'genetic-deepseek-mbpp',
    22: 'genetic-sonnet3.7-mbpp',
}
print(__name__)
if __name__ == '__main__':
    experiment_id = int(os.getenv('experiment'))
    human_eval_instances = json.loads(os.getenv('human_eval_instances'))
    print(human_eval_instances)
    experiment_to_run = experiments[experiment_id]
    file_name = f'output/{experiment_to_run}.txt'
    print(f'Running experiment: {experiment_to_run}')
    print(f'The output file is {file_name}')
    orig_stdout = sys.stdout
    f = open(file_name, 'w')
    sys.stdout = f
    # sys.stdout = orig_stdout
    if experiment_id == 1:
        raise NotImplementedError
        # MagicCoderRunner().run_experiment_llama70(human_eval_instances)
    elif experiment_id == 2:
        raise NotImplementedError
        # CodellamaExperiments().run_experiment(human_eval_instances)
    elif experiment_id == 3:
        MagicCoderRunner().run_experiments_gensim(instances=human_eval_instances)
    elif experiment_id == 4:
        CodellamaExperiments().run_experiments_gensim(instances=human_eval_instances)
    elif experiment_id == 5:
        raise NotImplementedError
        # MagicCoderRunner().run_experiment_llama7(human_eval_instances)
    elif experiment_id == 6:
        GPTRunner().run_experiment_gensim(instances=human_eval_instances, population_size=5, dataset_choice=1, mutation_tool=1, experiment_to_run=experiment_to_run)
    elif experiment_id == 7:
        GPTRunner().run_experiment_gensim(instances=human_eval_instances, population_size=8,dataset_choice=3,mutation_tool=1,experiment_to_run=experiment_to_run)
    elif experiment_id == 8:
        results = []
        random_choices = random.choices(range(100000), k=10)
        for i in random_choices:
            results.append(GPTRunner().run_experiment_gensim(instances=human_eval_instances, population_size=5, seed=i))
            print(results)
        print(results)
        print(f'the average pass@1 is: {sum(results)/len(results)}')
    elif experiment_id == 9:
        GPTRunner().run_experiment_gensim(instances=None, population_size=5, dataset_choice=2, experiment_to_run=experiment_to_run, mutation_tool=1)
    elif experiment_id == 10:
        GPTRunner().run_experiment_gensim(instances=human_eval_instances, population_size=5, mutation_tool=2)
    elif experiment_id == 11:
        results = []
        random_choices = random.choices(range(100000), k=10)
        for i in random_choices:
            results.append(GPTRunner().run_experiment_gensim(instances=None, population_size=5, seed=i, dataset_choice=2))
            print(results)
        print(results)
        print(f'the average pass@1 is: {sum(results) / len(results)}')
        print("Invalid experiment")
    elif experiment_id == 12:
        LlamaExperiments().run_experiment_gensim(instances=human_eval_instances, population_size=5, mutation_tool=1, dataset_choice=1)
    elif experiment_id == 13:
        LlamaExperiments().run_experiment_gensim(instances=human_eval_instances, population_size=5, dataset_choice=1, use_deep_seek=True)
    elif experiment_id == 14:
        GPTRunner().run_experiment_gensim(instances=human_eval_instances, population_size=5, mutation_tool=1, dataset_choice=3, experiment_to_run=experiment_to_run)
    elif experiment_id == 15:
        GPTRunner().run_experiment_gensim(instances=human_eval_instances, population_size=5, mutation_tool=2, dataset_choice=3, experiment_to_run=experiment_to_run)
    elif experiment_id == 16:
        GPTRunner().run_experiment_gensim(instances=human_eval_instances, population_size=5, mutation_tool=2,
                                          dataset_choice=3, experiment_to_run=experiment_to_run)
    elif experiment_id == 17:
        AntropicRunner().run_experiment_gensim(population_size=5, mutation_tool=1, dataset_choice=3, experiment_to_run=experiment_to_run)
    elif experiment_id == 18:
        FireworksExperiments().run_experiment_gensim(population_size=5, mutation_tool=1, dataset_choice=3, experiment_to_run=experiment_to_run)
    elif experiment_id == 19:
        FireworksExperiments().run_experiment_gensim(population_size=5, mutation_tool=1, dataset_choice=1, experiment_to_run=experiment_to_run)
    elif experiment_id == 20:
        AntropicRunner().run_experiment_gensim(population_size=5, mutation_tool=1, dataset_choice=1, experiment_to_run=experiment_to_run)
    elif experiment_id == 21:
        FireworksExperiments().run_experiment_gensim(population_size=5, mutation_tool=1, dataset_choice=2, experiment_to_run=experiment_to_run)
    elif experiment_id == 22:
        AntropicRunner().run_experiment_gensim(population_size=5, mutation_tool=1, dataset_choice=2, experiment_to_run=experiment_to_run)
    f.close()
