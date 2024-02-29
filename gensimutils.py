import random
import time

import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from gensim.models import Word2Vec
import gensim.downloader
glove_vectors = gensim.downloader.load('fasttext-wiki-news-subwords-300')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')


def fetch_pos_identity(pos_tag):
    '''
    This method returns

    1. 'np' for proper nouns, 'n' for all other nouns

    2. 'a' for adjectives

    3. 'v' for verbs

    4. 'r' for adverbs

    5. None for all other tags
    '''

    if pos_tag in ['NNP', 'NNPS']:
        return 'np'
    elif pos_tag in ['NN', 'NNS']:
        return 'n'
    elif pos_tag in ['JJ', 'JJR', 'JJS']:
        return 'a'
    elif pos_tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        return 'v'
    elif pos_tag in ['RB', 'RBR', 'RBS']:
        return 'r'
    else:
        return None


def get_related_words(word, pos_tag, similarity_threshold):
    '''
    This method returns most similar words to the word passed.

    args:

    word = input word
    pos_tag = Simple POS tag of the word
    similarity_threshold (float) = Value between 0 and 1. Indicates the similarity threshold to consider

    returns:

    a list of similar words, along with the original word
    '''

    # Lemmatize the word
    word = lemmatizer.lemmatize(word, pos_tag)
    # Get the synonyms and antonyms of a word
    synonyms = [word]
    # antonyms = []

    try:
        vector_check = glove_vectors.get_vector(word)
    except:
        # If the word does not exist in the Glove model, return
        return synonyms

    for syn in wordnet.synsets(word):

        for l in syn.lemmas():

            try:

                if l.name() in synonyms:
                    continue

                # Get the vector of the synonym
                vector_prospect = glove_vectors.get_vector(l.name())

                # print('Checking word = ', l.name())
                cosine_diff = glove_vectors.cosine_similarities(vector_1=vector_check, vectors_all=[vector_prospect])
                # print(cosine_diff)

                # similar_by_vector()words_closer_than()n_similarity()
                if cosine_diff > similarity_threshold:
                    synonyms.append(l.name())

            except:

                pass

            # if l.antonyms():
            #   antonyms.append(l.antonyms()[0].name())

    return synonyms


def get_next_position(total_synonym_array, position_array, last_position):
    '''

    This method returns the next position of word replacement.

    args:

    total_synonym_array = Array containing the total length of synonyms
    position_array = Array containing current positions
    last_position_array = Integer

    returns:

    next position to be updated, -1 if all positions are exhausted
    '''
    new_pos = last_position

    for i in range(len(total_synonym_array)):

        # get a new position
        new_pos = (new_pos + 1) % len(total_synonym_array)

        # if the new position is not good enough, fetch a new one
        if position_array[new_pos] == -1 or position_array[new_pos] == total_synonym_array[new_pos]:
            continue
        else:
            return new_pos

    return -1


def provide_alternate_sentence(sentence, num_versions=1, max_changes=1, similarity_threshold=0.7, ignore_stopwords=True,
                               ignore_proper_nouns=True, mutation_probability=0.4):
    '''
    This method returns an alternate version(s) of the sentence passed by replacing words with their closest synonyms.

    args:

    sentence (String) = the input sentence
    num_versions (int) = the number of alternate versions required
    max_changes (int) = the maximum number of changes between versions
    similarity_threshold (float) = Value between 0 and 1. Indicates the similarity threshold to consider while replacing words
    ignore_stopwords (bool) = If True, stopwords will not be considered for replacement
    ignore_proper_nouns (bool) = If True, proper nouns will be ignored for replacement

    returns:

    list of alternate sentence(s)
    '''

    alternate_sentences = []

    sentence_combination = []

    # split the sentence into words
    words = sentence.split()

    # pos tag the sentence
    pos_tags = nltk.pos_tag(words)

    for each_word_pos in pos_tags:

        word = each_word_pos[0]
        pos_tag = each_word_pos[1]
        short_pos = fetch_pos_identity(pos_tag)

        # ignore proper nouns
        if ignore_proper_nouns and 'np' == short_pos:
            sentence_combination.append([word])
            continue

        # lemmatize the word
        if short_pos is not None:
            word_lemmatized = lemmatizer.lemmatize(word, short_pos)
        else:
            word_lemmatized = lemmatizer.lemmatize(word)

        # ignore stopwords
        if ignore_stopwords and (word_lemmatized in stop_words or word in stop_words):
            sentence_combination.append([word])
            continue

        # if POS is noun, adj, adv, or verb - get similar words
        if short_pos is not None:
            sentence_combination.append(get_related_words(word, short_pos, similarity_threshold))
        # else do nothing
        else:
            sentence_combination.append([word])
            continue

    for i in range(num_versions):

        alt_sentence = ''
        for j in sentence_combination:
            alt_sentence = alt_sentence + ' '
            if len(j) == 1:  ## no other choices for the word
                alt_sentence = alt_sentence + j[0]
            else:  ## there are other choices
                if random.random() > mutation_probability:  ## no mutation
                    alt_sentence = alt_sentence + j[0]
                else:
                    chosen_word = random.choices(j[1:], k=1)
                    alt_sentence = alt_sentence + chosen_word[0]

        alt_sentence = alt_sentence.strip()
        alternate_sentences.append(alt_sentence)

    return alternate_sentences


import re

special_token = '"""#SPECIAL_TOKEN'


def generate_arg_text(arg1=None, arg2=None):
    is_set = 'set' in arg2.lower()
    is_list = 'list' in arg2.lower()
    if arg1 is not None:
        temp1 = f'- {arg1}:'
    else:
        temp1 = ''
    is_float = 'float' in arg2.lower()
    is_bool = 'bool' in arg2.lower()
    is_int = 'int' in arg2.lower()
    is_number = is_float or is_int
    is_string = 'str' in arg2.lower()
    if is_set:
        temp1 += 'a set of '
        if is_number:
            if is_int:
                temp1 += 'integer numbers'
            elif is_float:
                temp1 += 'floting point numbers'
        elif is_string:
            temp1 += 'strings'
        elif is_bool:
            temp1 += 'booleans'
    elif is_list:
        temp1 += 'a list of '
        if is_number:
            if is_int:
                temp1 += 'integer numbers'
            elif is_float:
                temp1 += 'floting point numbers'
        elif is_string:
            temp1 += 'strings'
        elif is_bool:
            temp1 += 'booleans'
    else:
        temp1 += ''
        if is_number:
            if is_int:
                temp1 += 'an integer number'
            elif is_float:
                temp1 += 'a floting point number'
        elif is_string:
            temp1 += 'a string'
        elif is_bool:
            temp1 += 'a boolean'
    return temp1


def refactor_prompt(prompt):
    method_name = re.findall('def .*\(', prompt)[0].replace('def ', '').replace('(', '')
    try:
        arguments = re.findall('\(.*\)', prompt)[0].replace(')', '').replace('(', '').split(',')
    except IndexError:
        return ''
    try:
        returns = re.findall('\(.*\) -> .*:', prompt)[0].split('->')[1].replace(':', '')
    except IndexError:
        returns = ''
    args_list = []
    for arg in arguments:
        args_list.append(arg.split(":"))

    function_args_text = ''
    for arg in args_list:
        if len(arg) != 2:
            continue
        temp1 = generate_arg_text(arg1=arg[0], arg2=arg[1])
        function_args_text += f'{temp1} \n'

    function_return_text = generate_arg_text(arg1=None, arg2=returns)

    text = f'the {method_name} function takes {len(args_list)} parameters: \n' + function_args_text + f'the {method_name} function returns: \n' + function_return_text + ' \n '
    prompt.split(special_token)

    return text


def produce_first_generation():
    from datasets import load_dataset
    a = time.time()
    human_eval = load_dataset("openai_humaneval")
    human_eval_processed = []
    special_token = '"""#SPECIAL_TOKEN'
    import re
    for index, a_test in enumerate(human_eval['test']):
        a_test = a_test['prompt']
        a_test = a_test.replace("'''", '"""').replace('"""', special_token)
        exp = [m.start() for m in re.finditer(special_token, a_test)]
        if len(exp) == 1:
            a_test += special_token
        exp = [m.start() for m in re.finditer(special_token, a_test)]
        if len(exp) != 2:
            print(index)
        human_eval_processed.append(a_test)

    ignore_refactor = [10, 32, 38, 50, 64]
    first_generation_prompts_refactored = []
    for index, a_test in enumerate(human_eval_processed):
        # print(index)
        # print(a_test)
        if index not in ignore_refactor:
            added_text = refactor_prompt(a_test)
        splits = a_test.split(special_token)
        if len(splits) != 5:
            alternate_sentences = provide_alternate_sentence(splits[1], num_versions=4, similarity_threshold=0.5)
            final_sentence = [splits[0] + special_token + '\n ' + added_text + alternative + special_token + splits[2]
                              for alternative in alternate_sentences]
        else:
            alternate_sentences1 = provide_alternate_sentence(splits[1], num_versions=4, similarity_threshold=0.5)
            alternate_sentences2 = provide_alternate_sentence(splits[3], num_versions=4, similarity_threshold=0.5)
            final_sentence = [
                splits[0] + special_token + '\n' + alternative[0] + special_token + splits[2] + special_token +
                alternative[1] + special_token + splits[4] for alternative in
                zip(alternate_sentences1, alternate_sentences2)]
        final_list = [a_test] + final_sentence
        first_generation_prompts_refactored.append(final_list)
    print('total time')
    print(time.time() - a)
    print(first_generation_prompts_refactored)