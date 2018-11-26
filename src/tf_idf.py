import gensim
import re
from collections import Counter
from clean_data import clean_data_directories, load_raw_reports
from math import log
def calc_idf():
    doc_counter = Counter()
    doc_count = 0
    for clean_dir in clean_data_directories:
        reports, _ = load_raw_reports(clean_dir)
        for report in reports:
            doc_count +=1
            for word in set(filter(lambda x: x != "\n", report.lower().split())):
                doc_counter[word] += 1
    for word in doc_counter:
        doc_counter[word] = log(1 + doc_count / doc_counter[word])
    return dict(doc_counter)


def calc_tf(word2vec_model, report):
    # Given a report, for each word in the report calculates the term frequency
    words = filter(lambda x: x != "\n", report.lower().split())
    return Counter(words)


