import os
import json
# os.environ['TRANSFORMERS_CACHE'] = '/home/hamedth/projects/def-hemmati-ac/hamedth/hugging_face'
from magiccoder_experiments import MagicCoderRunner
from gpt_experiments import GPTRunner
from dotenv import load_dotenv

load_dotenv()
from codellama_experiments import CodellamaExperiments
experiments = {
    1: 'genetic-magiccoder-llama2-70b',
    2: 'genetic-codellama-llama2-70b',
    3: 'genetic-magiccoder-gensim',
    4: 'genetic-magiccoder-gensim-original-testcases',
    5: 'genetic-codellama-gensim',
    6: 'baseline-magicoder',
    7: 'baseline-codellama',
    8: 'genetic-magicoder-llama2-7b',
    9: 'genetic-gpt4-gensim-generated-testcases',
    10: 'genetic-gpt4-gensim-original-testcases-ten_population',
    11: 'genetic-gpt4-gensim-original-testcases-v2',
    12: 'genetic-gpt4-gensim-generated-testcases'
}
print(__name__)
if __name__ == '__main__':
    experiment_id = int(os.getenv('experiment'))
    human_eval_instances = json.loads(os.getenv('human_eval_instances'))
    print(human_eval_instances)
    experiment_to_run = experiments[experiment_id]
    print(f'Running experiment: {experiment_to_run}')
    if experiment_id == 1:
        MagicCoderRunner().run_experiment_llama70(human_eval_instances)
    elif experiment_id == 2:
        CodellamaExperiments().run_experiment(human_eval_instances)
    elif experiment_id == 3:
        MagicCoderRunner().run_experiments_gensim(instances=human_eval_instances, first_generation_openai=True, with_original_testcases=False)
    elif experiment_id == 4:
        MagicCoderRunner().run_experiments_gensim(instances=human_eval_instances, first_generation_openai=True, with_original_testcases=True)
    elif experiment_id == 5:
        CodellamaExperiments().run_experiments_gensim(instances=human_eval_instances, first_generation_openai=True, with_original_testcases=True)
    elif experiment_id == 6:
        pass
    elif experiment_id == 7:
        pass
    elif experiment_id == 8:
        MagicCoderRunner().run_experiment_llama7(human_eval_instances)
    elif experiment_id == 9:
        GPTRunner().run_experiment_gensim(first_generation_openai=True, instances=human_eval_instances, with_original_testcases=False, population_size=5)
    elif experiment_id == 10:
        GPTRunner().run_experiment_gensim(first_generation_openai=True, instances=human_eval_instances, with_original_testcases=True, population_size=10)
    elif experiment_id == 11:
        GPTRunner().run_experiment_gensim(first_generation_openai=True, instances=human_eval_instances, with_original_testcases=True, population_size=5, version=2)
    elif experiment_id == 12:
        GPTRunner().run_experiment_gensim(first_generation_openai=True, instances=None, with_original_testcases=False, population_size=5, dataset_choice=2)
    else:
        print("Invalid experiment")
