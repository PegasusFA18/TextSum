from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import sys
import io
import os

def summarize(file):
	from skipthoughts import encoder
	data = read_report(file)
	sentences = sent_tokenize(data)
	#model = skipthoughts.load_model()
	#encoder = skipthoughts.Encoder(model)
	vectors = encoder.encode(sentences, verbose=False)
	n = int(len(vectors)**0.5)
	print('N:', n)
	print('Sentences:', len(vectors))
	print('Vector size:', len(vectors[0]))
	kmeans = KMeans(n_clusters=n).fit(vectors)
	centroids = kmeans.cluster_centers_
	print('Centroids:', centroids)
	print(centroids.shape)
	clusters = [[] for i in range(n)]
	for i in range(len(vectors)):
		label = kmeans.labels_[i]
		clusters[label].append(i)
	avg_order, min_dist = [], []
	print('Clusters:', clusters)
	for i in range(n):
		avg_order.append(np.mean(clusters[i]))
		argmin, distances = pairwise_distances_argmin_min(centroids[i:i+1], np.array([vectors[j] for j in clusters[i]]))	
		min_dist.append(clusters[i][argmin[0]])
	print('Avg order', avg_order)
	print('Min dist', min_dist)
	order = sorted(list(range(n)), key=lambda x: avg_order[x])
	print('Cluster Order', order)
	summary = ''
	for i in order:
		summary += sentences[min_dist[i]] + '\n'
	rand_summ = ''
	for i in order:
		rand_summ += sentences[clusters[i][np.random.randint(0, len(clusters[i]))]] + '\n'
	print('Random summary: ' + rand_summ)
	write_summary(file, summary)
	return summary

def write_summary(filename, summary):
	splits = filename.split('cleaned')
	with io.open(splits[0] + 'summaries'+ splits[1] +'summary.txt', 'w+', encoding='utf-8') as f:
		f.write(summary)	

def read_report(filename):
	data = None
	with io.open(filename, 'r', encoding='utf-8') as file:
		data = file.read()
	return data

if __name__ == '__main__':
	arg = sys.argv[1]
	if arg.endswith('.txt'):
		summarize(arg)
	elif False:
		for root, dirs, files in os.walk('../data/cleaned/'):
			for file in files:
				if '.txt' in file:
					summary = summarize(read_report(root+'/'+file))
					write_summary(root+'/'+file, summary)