import os
os.environ['TRANSFORMERS_CACHE'] = '/home/hamedth/projects/def-hemmati-ac/hamedth/hugging_face'
from magiccoder_experiments import MagicCoderRunner
from dotenv import load_dotenv

load_dotenv()
from codellama_experiments import CodellamaExperiments
experiments = {
    1: 'genetic-magiccoder-llama2-70b',
    2: 'genetic-codellama-llama2-70b',
    3: 'genetic-magiccoder-gensim',
    4: 'genetic-codellama-gensim',
    5: 'baseline-magicoder',
    6: 'baseline-codellama',
    7: 'genetic-magicoder-llama2-7b'
}
print(__name__)
if __name__ == '__main__':
    experiment_id = int(os.getenv('experiment'))
    experiment_to_run = experiments[experiment_id]
    print(f'Running experiment: {experiment_to_run}')
    if experiment_id == 1:
        MagicCoderRunner().run_experiment_llama70()
    elif experiment_id == 2:
        CodellamaExperiments().run_experiment()
    elif experiment_id == 3:
        MagicCoderRunner().run_experiments_gensim()
    elif experiment_id == 4:
        CodellamaExperiments().run_experiments_gensim()
    elif experiment_id == 5:
        pass
    elif experiment_id == 6:
        pass
    elif experiment_id == 7:
        MagicCoderRunner().run_experiment_llama7()
    else:
        print("Invalid experiment")
