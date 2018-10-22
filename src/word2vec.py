# A file that trains word2vec on our cleaned dataset, and saves the trained model in models/word2vec
import gensim
import logging
import nltk
import os
import time
import argparse
from clean_data import clean_data_directories, load_raw_reports

word2vec_model_dir = '../models/word2vec/'


def generate_corpus():
    # Generates a corpus in the form of a list of sentences, where each sentence is now a list of words instead of a string
    list_of_word_lists = []
    for clean_dir in clean_data_directories:
        reports, names = load_raw_reports(clean_dir)
        for report in reports:
            for sentence in report.split('\n'):
                list_of_word_lists.append(sentence.split())
    return list_of_word_lists
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None)
    args = parser.parse_args()
    model_name = args.model_name if args.model_name else 'model_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    corpus = generate_corpus()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Word2Vec(corpus, min_count=1)
    os.makedirs(os.path.dirname(word2vec_model_dir+"test.txt"), exist_ok=True)
    model.save(word2vec_model_dir + model_name)



if __name__ == '__main__':
    main()
