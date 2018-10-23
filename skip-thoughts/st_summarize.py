import skipthoughts
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import sys

def summarize(data):
	sentences = sent_tokenize(data)
	model = skipthoughts.load_model()
	encoder = skipthoughts.Encoder(model)
	vectors = encoder.encode(sentences)
	n = int(len(vectors)**0.5)
	kmeans = KMeans(n_clusters=n).fit(vectors)
	summary = []
	centroids = kmeans.cluster_centers_
	clusters = np.array([[] for i in range(n)])
	print(clusters.shape)
	for i in range(len(vectors)):
		label = kmeans.labels_[i]
		clusters[label] = np.append(clusters[label], [i])
	avg_order = []
	for i in range(n):
		avg_order.append(np.mean(clusters[i]))
	argmin, distances = pairwise_distances_argmin_min(np.reshape(centroids, (n,1)), np.array([[vectors[i] for i in clusters[center]] for center in range(n)]))	
	order = sorted(range(n), key=lambda x: avg_order[x])
	for i in order:
		summary.append(sentences[argmin[i]])
	return summary

def strip_text(text):
	return text

if __name__ == '__main__':
	summarize(sys.argv[1])