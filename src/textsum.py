import gensim
import argparse
import os
from clean_data import clean_data_directories, load_raw_reports
from sklearn.cluster import MiniBatchKMeans
from word2vec import word2vec_model_dir
import numpy as np
from lexrank import STOPWORDS, LexRank
from itertools import chain

summary_directories = ['../summaries/annual/', '../summaries/quarterly/']
google_word2vec_dir = '../models/pretrained_word2vec/GoogleNews-vectors-negative300.bin'
google_word2vec_load_limit = 800000

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
            print("Using Google's pretrained model")
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


def summarize_report(report, model, model_type):
    summary_methods = {
        "word2vec": summarize_report_word2vec, 
        "lexrank": summarize_report_lexrank
    }
    return summary_methods[model_type](report, model)


def summarize_report_word2vec(report, model):
    sentence_vectors = []
    report = report.split('\n')
    vectorizable_report = []
    words_in_vocab = 0
    total_words = 0
    for sentence in report:
        words = sentence.split()
        word_vectors = [model.wv[word] for word in words if word in model.wv.vocab]
        words_in_vocab += len(word_vectors)
        total_words += len(words)
        if len(word_vectors) == 0:
            print("DEBUG: ", sentence)
            continue
        sent_vec = np.average(np.stack(word_vectors), axis=0)
        sentence_vectors.append(sent_vec)
        vectorizable_report.append(sentence)
    print("Number of total words in report :", total_words)
    print("Number of words found in vocab : ", words_in_vocab)
    X = np.array(sentence_vectors)
    if len(sentence_vectors) < 20:
        num_clusters = 1
    else:
        num_clusters = 20
    means = MiniBatchKMeans(n_clusters=num_clusters,batch_size=6,max_iter=10).fit(X)
    summary = []
    summary_sent_idx = []
    for center in means.cluster_centers_:
        best_sentence = 0
        best_distance = float('inf')
        for i in range(len(sentence_vectors)):
            curr_distance = np.linalg.norm(center - sentence_vectors[i])
            if curr_distance < best_distance and vectorizable_report[i] not in summary:
                best_sentence = i
                best_distance = curr_distance
        summary.append(vectorizable_report[best_sentence].capitalize())
        summary_sent_idx.append(best_sentence)
    sorted_summary = [x for _,x in sorted(zip(summary_sent_idx,summary))]
    return '\n'.join(sorted_summary)


def main():
    model, model_type = load_model()
    if model is None:
        return
    for clean_dir, summary_dir in zip(clean_data_directories, summary_directories):
        reports, names = load_raw_reports(clean_dir)
        summaries = [summarize_report(report, model, model_type) for report in reports]
        summary_dir = summary_dir + model_type + "/"
        os.makedirs(os.path.dirname(summary_dir+"test.txt"), exist_ok=True)
        for summary, name in zip(summaries, names):
            cleaned_name = name.replace('_cleaned.txt', '_summary.txt')
            f = open(summary_dir+cleaned_name, 'w+')
            f.write(summary)
            f.close()
    
if __name__ == '__main__':
    main()