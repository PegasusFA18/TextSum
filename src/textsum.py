from gensim.models import KeyedVectors, Word2Vec
import argparse
import os
from clean_data import clean_data_directories, load_raw_reports
from sklearn.cluster import KMeans
from word2vec import word2vec_model_dir
import numpy as np
from lexrank import STOPWORDS, LexRank
import logging
from nltk.tokenize import sent_tokenize
from tf_idf import calc_tf, calc_idf
from math import log


# IMPORTANT CONSTANTS
# (1) Locations
SUMMARY_DIRECTORIES = ['../summaries/annual/', '../summaries/quarterly/']
# (2) Word2Vec specific
GOOGLE_WORD2VEC_DIR = '../models/pretrained_word2vec/GoogleNews-vectors-negative300.bin'
GOOGLE_WORD2VEC_LOAD_LIMIT = 800000
WORD2VEC_NUM_CLUSTERS = 2
WORD2VEC_MIN_SENTENCE_LENGTH = 4
HEADER_LENGTH_LIMIT = 8
# (3) LexRank specific
LEXRANK_SUMMARY_SIZE = 10
# (4) Logging-related
logging.basicConfig(level=logging.DEBUG)


def main():
    model, model_type = load_model()
    summary_len_total = 0
    report_len_total = 0
    worst_reduction_ratio = 0
    best_reduction_ratio = 1
    for clean_dir, summary_dir in zip(clean_data_directories, SUMMARY_DIRECTORIES):
        model_summary_dir = summary_dir + model_type + "/"
        reports, names = load_raw_reports(clean_dir)
        os.makedirs(os.path.dirname(model_summary_dir), exist_ok=True)
        for report, name in zip(reports, names):
            summary = summarize_report(report, model, model_type)
            # Summary statistics related
            summary_len_total += len(summary)
            report_len_total += len(report)
            ratio = len(summary) / len(report)
            worst_reduction_ratio = max(ratio, worst_reduction_ratio)
            best_reduction_ratio = min(ratio, best_reduction_ratio)
            # File writing
            summary_file_name = name.replace('_cleaned.txt', '_summary.txt')
            f = open(model_summary_dir + summary_file_name, 'w+')
            f.write(summary)
            f.close()
    avg_reduction = summary_len_total/report_len_total
    print("Average reduction ", avg_reduction)
    print("Best reduction ratio ", best_reduction_ratio)
    print("Worst reduction ratio ", worst_reduction_ratio)


def load_model():
    # Based on command line args, returns model and type
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--model_name', type=str, default=None)
    args = parser.parse_args()
    if args.model_type == 'word2vec':
        if args.model_name is not None:
            model = Word2Vec.load(word2vec_model_dir + args.model_name)
            return model, 'word2vec'
        else:
            logging.info("Using Google's pretrained model")
            model = KeyedVectors.load_word2vec_format(GOOGLE_WORD2VEC_DIR,
                                                      binary=True,
                                                      limit=GOOGLE_WORD2VEC_LOAD_LIMIT)
            return model, 'word2vec'
    elif args.model_type == 'lexrank':
        cleaned_reports = [load_raw_reports(clean_dir)[0] for clean_dir in clean_data_directories]
        lxr = LexRank(cleaned_reports, STOPWORDS['en'])
        return lxr, 'lexrank'
    else:
        print('Default baseline - first sentence')
        return None, "first_sentence"
        


def summarize_report(report, model, model_type):
    summary_methods = {
        "word2vec": summarize_report_word2vec,
        "lexrank": summarize_report_lexrank,
        "first_sentence": summarize_report_first_sent
    }
    return summary_methods[model_type](report, model)


def summarize_report_lexrank(report, model):
    sentences = report.split('\n')
    summary = model.get_summary(sentences, summary_size=10)
    return '\n'.join([sentence.capitalize() for sentence in summary])


def summarize_report_first_sent(report, model):
    paragraphs, headers, pointers = extract_paragraphs_and_headers(report)
    paragraph_summaries = [paragraph[0] for paragraph in paragraphs]
    return combine_paragraph_summaries_and_headers(paragraph_summaries, headers, pointers)


def combine_paragraph_summaries_and_headers(paragraph_summaries, headers, pointers):
    if len(headers) == 0:
        # No headers in this report
        return '\n'.join([''.join(summary) for summary in paragraph_summaries])
    else:
        summary = []
        last_header_ptr = -1
        for p_idx in range(len(pointers)):
            ptr = pointers[p_idx]
            while ptr > last_header_ptr:
                last_header_ptr += 1
                summary.extend([headers[last_header_ptr], '\n\n'])
            summary.append(''.join(paragraph_summaries[p_idx]))
            summary.append('\n\n')
        return ''.join(summary)


def summarize_report_word2vec(report, model):
    tf = calc_tf(model, report)
    idf = calc_idf()
    paragraphs, headers, pointers = extract_paragraphs_and_headers(report)
    paragraph_summaries = []
    for paragraph in paragraphs:
        sentence_vectors, vectorizable_paragraph = vectorize_paragraph(paragraph, model, tf, idf)
        if len(vectorizable_paragraph) == 0:
            # None of the sentences were vectorizable, so we choose the first sentence
            paragaph_summary = paragraph[0]
            paragraph_summaries.append([paragraph[0]])
        else:
            paragaph_summary = cluster_and_summarize(sentence_vectors, vectorizable_paragraph)
        paragraph_summaries.append(paragaph_summary)
    return combine_paragraph_summaries_and_headers(paragraph_summaries, headers, pointers)


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


def vectorize_paragraph(paragraph, model, tf, idf):
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
        word_vectors = [model[word.lower()] for word in words if word.lower() in model.vocab]
        word_tf = [tf[word.lower()] for word in words if word.lower() in model.vocab]
        word_idf = [idf[word.lower()] for word in words if word.lower() in model.vocab]
        word_weights = [((1 + log(t)) * i) for t, i in zip(word_tf, word_idf)]
        word_vectors = [vector*weight for vector, weight in zip(word_vectors, word_weights)]
        if len(word_vectors) == 0:
            logging.debug("Unable to vectorize following sentence : %s " % (sentence))
            continue
        sent_vec = np.average(np.stack(word_vectors), axis=0)
        sentence_vectors.append(sent_vec)
        vectorizable_paragraph.append(sentence)
        words_in_vocab += len(word_vectors)
        total_words += len(words)
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
    sorted_summary = [x for _, x in sorted(zip(summary_sent_idx, summary))]
    return sorted_summary


if __name__ == '__main__':
    main()
