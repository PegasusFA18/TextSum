import gensim
import argparse
import os
from clean_data import clean_data_directories, load_raw_reports
from sklearn.cluster import MiniBatchKMeans
from word2vec import word2vec_model_dir
import numpy as np

summary_directories = ['../summaries/annual/', '../summaries/quarterly/']

def load_word2vec_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='word2vec')
    parser.add_argument('--model_name', type=str, default=None)
    args = parser.parse_args()
    if args.model_type == 'word2vec':
        if args.model_name is not None:
            model = gensim.models.Word2Vec.load(word2vec_model_dir+args.model_name)
            return model
        else:
            print("Need model name!")
            return
    else:
        print('Other models coming soon!')
        return

def summarize_report(report, model):
    sentence_vectors = []
    report = report.split('\n')
    for sentence in report:
        word_vectors = np.stack(model.wv[word] for word in sentence.split())
        sent_vec = np.average(word_vectors, axis=0)
        sentence_vectors.append(sent_vec)
    X = np.array(sentence_vectors)
    if len(sentence_vectors) < 5:
        num_clusters = 1
    else:
        num_clusters = 5
    means = MiniBatchKMeans(n_clusters=num_clusters,batch_size=6,max_iter=10).fit(X)
    summary = []
    for center in means.cluster_centers_:
        best_sentence = 0
        best_distance = float('inf')
        for i in range(len(sentence_vectors)):
            curr_distance = np.linalg.norm(center - sentence_vectors[i])
            if curr_distance < best_distance and report[best_sentence] not in summary:
                best_sentence = i
                best_distance = curr_distance
        summary.append(report[best_sentence])
    return '\n'.join(summary)


def main():
    model = load_word2vec_model()
    if model is None:
        return
    for clean_dir, summary_dir in zip(clean_data_directories, summary_directories):
        reports, names = load_raw_reports(clean_dir)
        summaries = [summarize_report(report, model) for report in reports]
        os.makedirs(os.path.dirname(summary_dir+"test.txt"), exist_ok=True)
        for summary, name in zip(summaries, names):
            cleaned_name = name.replace('_cleaned.txt', '_summary.txt')
            f = open(summary_dir+cleaned_name, 'w+')
            f.write(summary)
            f.close()
    
if __name__ == '__main__':
    main()