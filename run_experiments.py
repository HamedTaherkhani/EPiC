import os
os.environ['TRANSFORMERS_CACHE'] = '/home/hamedth/projects/def-hemmati-ac/hamedth/hugging_face'
from magiccoder_experiments import MagicCoderRunner

experiments = {
    1: 'genetic-magiccoder-llama2',
    2: 'genetic-codellama-llama2',
    3: 'genetic-magiccoder-gensim',
    4: 'genetic-codellama-gensim',
    5: 'baseline-magicoder',
    6: 'baseline-codellama',

}
# experiment_number = input(f'What experiment do you want to run? {str(experiments)}')
# if experiment_number == '1':
MagicCoderRunner().run_experiment()