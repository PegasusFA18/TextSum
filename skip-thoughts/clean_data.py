# -*- coding: utf-8 -*-
# coding: utf-8

# A script that generates a folder data/cleaned
from __future__ import unicode_literals
from nltk.tokenize import sent_tokenize
import os
from collections import Counter
import nltk

undesired_phrases = ['DATA_TABLE_REMOVED', '| 2017 Form 10-K |']
undesired_characters = ['®', '™']
raw_data_directories = ['../data/raw/annual/', '../data/raw/quarterly/']
clean_data_directories = ['../data/cleaned/annual/', '../data/cleaned/quarterly/']

def filter_sentence(sentence):
    for phrase in undesired_phrases:
        if phrase in sentence:
            return False
    return True

def clean_sentence(sentence):
    ret = sentence
    for char in undesired_characters:
        ret = ret.replace(char, '')
    return ret

def sentencify_report(report):
    paragraphs = report.split('\n')
    sentences = []
    for paragraph in paragraphs:
        paragraph_sentences = sent_tokenize(paragraph)
        filtered_sentences = [clean_sentence(sentence) for sentence in paragraph_sentences if filter_sentence(sentence)]
        sentences.extend(filtered_sentences)
    return sentences

def clean_data(reports):
    sentencified_reports = [sentencify_report(report) for report in reports]
    return ['\n'.join(sentences) for sentences in sentencified_reports]

def load_raw_reports(raw_dir):
    reports = []
    names = []
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            if '.txt' in f:
                curr_report = open(raw_dir + f)
                reports.append(curr_report.read())
                names.append(f)
                curr_report.close()
    return reports, names

def save_cleaned_reports(reports, names, clean_dir):
    os.makedirs(os.path.dirname(clean_dir+"test.txt"), exist_ok=True)
    for report, name in zip(reports, names):
        cleaned_name = name.replace('.txt', '_cleaned.txt')
        f = open(clean_dir+cleaned_name, 'w+')
        f.write(report)
        f.close()

def main():
    for raw_dir, clean_dir in zip(raw_data_directories, clean_data_directories):
        reports, names = load_raw_reports(raw_dir)
        cleaned_reports = clean_data(reports)
        save_cleaned_reports(cleaned_reports, names, clean_dir)

if __name__ == '__main__':
    main()



