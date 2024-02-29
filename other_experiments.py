from gensimutils import produce_first_generation
from open_ai import generate_first_population_openai, produce_testcases
experiment = int(input(
    'Enter experiment: 1- first generation using gensim, 2- fist generation using openAI, 3- generating test cases using openAI'))
if experiment == 1:
    produce_first_generation()
elif experiment == 2:
    generate_first_population_openai()
elif experiment == 3:
    produce_testcases()