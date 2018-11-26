import gensim
import argparse
import os
from clean_data import clean_data_directories, load_raw_reports
from sklearn.cluster import KMeans
from word2vec import word2vec_model_dir
import numpy as np
from lexrank import STOPWORDS, LexRank
from itertools import chain
import logging
from nltk.tokenize import sent_tokenize
from time import time

### IMPORTANT CONSTANTS
summary_directories = ['../summaries/annual/', '../summaries/quarterly/']
google_word2vec_dir = '../models/pretrained_word2vec/GoogleNews-vectors-negative300.bin'
google_word2vec_load_limit = 800000
logging.basicConfig(level=logging.DEBUG)
LEXRANK_SUMMARY_SIZE = 10
WORD2VEC_NUM_CLUSTERS = 2
SUMMARIZE_LOG_EVERY_N = 1
WORD2VEC_MIN_SENTENCE_LENGTH = 4
HEADER_LENGTH_LIMIT = 8
###

def load_model():
    # Based on command line args, returns model and type 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--model_name', type=str, default=None)
    args = parser.parse_args()
    if args.model_type == 'word2vec':
        if args.model_name is not None:
            model = gensim.models.Word2Vec.load(word2vec_model_dir + args.model_name)
            return model, 'word2vec'
        else:
            logging.info("Using Google's pretrained model")
            model = gensim.models.KeyedVectors.load_word2vec_format(google_word2vec_dir,
                                                                    binary=True,
                                                                    limit=google_word2vec_load_limit)
            return model, 'word2vec'
    else:
        print('Default baseline - lexrank')
        cleaned_reports = [load_raw_reports(clean_dir)[0] for clean_dir in clean_data_directories]
        documents = list(chain(cleaned_reports))
        lxr = LexRank(cleaned_reports, STOPWORDS['en'])
        return lxr, 'lexrank'

def summarize_report_lexrank(report, model):
    sentences = report.split('\n')
    summary = model.get_summary(sentences, summary_size=10)
    return '\n'.join([sentence.capitalize() for sentence in summary])

def generate_sentence_vectors_word2vec(paragraph, model):
    # Given a paragraph, returns sentence_vectors and vectorizable_paragraph
    # (Vectorizable_report != num of sentences in paragraph)
    # because if none of the words in a sentence have a representation in the vocab,
    # we do not use it in the summary)
    sentence_vectors = []
    vectorizable_paragraph = []
    words_in_vocab = 0
    total_words = 0

    for sentence in paragraph:
        words = sentence.split()
        if len(words) < WORD2VEC_MIN_SENTENCE_LENGTH:
            continue
        word_vectors = [model.wv[word.lower()] for word in words if word.lower() in model.wv.vocab]
        if len(word_vectors) == 0:
            logging.debug(sentence)
            continue
        sent_vec = np.average(np.stack(word_vectors), axis=0)
        sentence_vectors.append(sent_vec)
        vectorizable_paragraph.append(sentence)
        words_in_vocab += len(word_vectors)
        total_words += len(words)
    #logging.info("Number of total words in report : %d"%(total_words))
    #logging.info("Number of words found in vocab : %d "%(words_in_vocab))
    #logging.info("Number of sentences that could not be vectorized : %d "%(
    #    len(paragraph) - len(vectorizable_paragraph)))
    return sentence_vectors, vectorizable_paragraph

def cluster_and_summarize(sentence_vectors, vectorizable_paragraph):
    # Given a list of sentence vectors and a list of corresponding sentences,
    # generates a summary using K-Means
    X = np.array(sentence_vectors)
    if len(sentence_vectors) < 5:
        centers = [np.average(X, axis=0)]
        num_clusters = 1
    else:
        num_clusters = len(sentence_vectors) // 5
        means = KMeans(n_clusters=num_clusters).fit(X)
        centers = means.cluster_centers_
    summary = []
    summary_sent_idx = []
    for center in centers:
        best_sentence = 0
        best_distance = float('inf')
        for i in range(len(sentence_vectors)):
            curr_distance = np.linalg.norm(center - sentence_vectors[i])
            if curr_distance < best_distance and vectorizable_paragraph[i] not in summary:
                best_sentence = i
                best_distance = curr_distance
        summary.append(vectorizable_paragraph[best_sentence])
        summary_sent_idx.append(best_sentence)
    sorted_summary = [x for _ , x in sorted(zip(summary_sent_idx,summary))]
    return sorted_summary

def extract_paragraphs_and_headers(report):
    """
    Given a report, returns three things:
    * A list of headers
    * A list of paragraphs
    * A list of pointers, s.t. pointers[i] represents the index of the header
    under which paragraphs[i] shows up (there could be paragraphs before or after it
    under that header)
    len(headers) >= len(paragraphs) == len(pointers) (each paragraph comes with a header)
    """
    headers = []
    paragraphs = []
    pointers = []
    curr_paragraphs = report.split("\n")
    for idx in range(len(curr_paragraphs)):
        curr_paragraph = sent_tokenize(curr_paragraphs[idx])
        if len(curr_paragraph) == 1 and curr_paragraph[0][-1] != ".":
            if not(curr_paragraph[0].isdigit()) and len(curr_paragraph[0].split()) < HEADER_LENGTH_LIMIT:
                # Not page number
                headers.append(curr_paragraph[0])
        else:
            paragraphs.append(curr_paragraph)
            if len(headers) == 0:
                # no header
                pointers.append(-1)
            else:
                pointers.append(len(headers) - 1)
    return paragraphs, headers, pointers

def summarize_report_word2vec(report, model):
    paragraphs, headers, pointers = extract_paragraphs_and_headers(report)
    paragraph_summaries = []
    for paragraph in paragraphs:
        sentence_vectors, vectorizable_paragraph = generate_sentence_vectors_word2vec(paragraph, model)
        if len(vectorizable_paragraph) == 0:
            # None of the sentences were vectorizable
            paragraph_summaries.append([paragraph[0]])
        else:
            paragraph_summaries.append(cluster_and_summarize(sentence_vectors, vectorizable_paragraph))
    if len(headers) == 0:
        # No headers
        return '\n'.join([''.join(summary) for summary in paragraph_summaries])
    else:
        summary = []
        last_header_ptr = -1
        for idx in range(len(paragraphs)):
            if pointers[idx] == -1:
                # No header(s) for this paragraph
                summary.append(''.join(paragraph_summaries[idx]))
                summary.append('\n')
            else:
                while(last_header_ptr != pointers[idx]):
                    last_header_ptr +=1
                    summary.append('\n')
                    summary.append(headers[last_header_ptr])
                    summary.append('\n\n')
                summary.append(''.join(paragraph_summaries[idx]))
                summary.append('\n')
        return ''.join(summary)


def summarize_report(report, model, model_type):
    summary_methods = {
        "word2vec": summarize_report_word2vec,
        "lexrank": summarize_report_lexrank
    }
    return summary_methods[model_type](report, model)

def main():
    model, model_type = load_model()
    if model is None:
        return
    for clean_dir, summary_dir in zip(clean_data_directories, summary_directories):
        reports, names = load_raw_reports(clean_dir)
        summaries = []
        start_time = time()
        summary_reductions = []
        for i in range(len(reports)):
            summary = summarize_report(reports[i], model, model_type)
            summary_len = len(summary)
            report_len = len(reports[i])
            summary_reductions.append(summary_len/report_len)
            if i % SUMMARIZE_LOG_EVERY_N == 0:
                curr_time = time()
                logging.info("Iteration %d : took %d seconds"%(i, curr_time - start_time))
                start_time = curr_time
            summaries.append(summary)
        summary_dir = summary_dir + model_type + "/"
        os.makedirs(os.path.dirname(summary_dir+"test.txt"), exist_ok=True)
        for summary, name in zip(summaries, names):
            cleaned_name = name.replace('_cleaned.txt', '_summary.txt')
            f = open(summary_dir+cleaned_name, 'w+')
            f.write(summary)
            f.close()
        avg_reduction = sum(summary_reductions)/len(summary_reductions)
        print("Average reduction ", avg_reduction)
    
if __name__ == '__main__':
    main()