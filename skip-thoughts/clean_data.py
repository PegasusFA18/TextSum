# A script that generates a folder data/cleaned
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
    if len(sentence.split(' ')) > 5:
        return True
    return False

def clean_sentence(sentence):
    ret = sentence
    for char in undesired_characters:
        ret = ret.replace(char, '')
    return ret

def clean_report(report):
    paragraphs = report.split('\n')
    clean_paragraphs = []
    for paragraph in paragraphs:
        paragraph_sentences = sent_tokenize(paragraph)
        filtered_sentences = [clean_sentence(sentence) for sentence in paragraph_sentences if filter_sentence(sentence)]
        if len(filtered_sentences) > 0:
            clean_paragraphs.append(' '.join(filtered_sentences))
    return clean_paragraphs

def clean_data(reports):
    paragraph_reports = [clean_report(report) for report in reports]
    return ['\n\n'.join(paragraphs) for paragraphs in paragraph_reports]

def load_raw_reports(raw_dir):
    reports = []
    names = []
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            if '.txt' in f:
                curr_report = open(raw_dir + f, 'r', encoding='utf8')
                reports.append(curr_report.read())
                names.append(f)
                curr_report.close()
    return reports, names

def save_cleaned_reports(reports, names, clean_dir):
    os.makedirs(clean_dir, exist_ok=True)
    for report, name in zip(reports, names):
        cleaned_name = name.replace('.txt', '_cleaned.txt')
        f = open(clean_dir+cleaned_name, 'w', encoding='utf8')
        f.write(report)
        f.close()

def main():
    for raw_dir, clean_dir in zip(raw_data_directories, clean_data_directories):
        reports, names = load_raw_reports(raw_dir)
        cleaned_reports = clean_data(reports)
        save_cleaned_reports(cleaned_reports, names, clean_dir)

if __name__ == '__main__':
    main()



