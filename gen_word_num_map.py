from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from collections import Counter
import pickle
def word2vec():
	def get_wav_files(wav_path):
		wav_files = []
		for (dirpath, dirname, filename) in os.walk(wav_path):
			for _filename in filename:
				if _filename.endswith('.wav') or _filename.endswith('.WAV'):
					filename_path = os.sep.join([dirpath, _filename])
					wav_files.append(filename_path)
		return wav_files

	def get_wav_label(wav_files, label_files):
		label_dict = {}
		with open(label_files, 'r', encoding='utf-8') as f:
			for label in f:
				label = label.strip('\n')
				label_id = label.split(' ', 1)[0]
				label_text = label.split(' ', 1)[1]
				label_dict[label_id] = label_text
		labels = []
		for wav_file in wav_files:
			wav_id = os.path.basename(wav_file).split('.')[0]
			if wav_id in label_dict:
				labels.append(label_dict[wav_id])
		return labels

	def get_vocabulary(labels):
		all_words = []
		for label in labels:
		    all_words += [word for word in label]
		counter = Counter(all_words)
		count_pairs = sorted(counter.items(), key=lambda x: -x[1])
		words, _ = zip(*count_pairs)

		word_num_map = dict(zip(words, range(len(words))))
		return word_num_map

	wav_path = './data/wav/train'
	label_files = './data/doc/trans/train.word.txt'
	wav_files = get_wav_files(wav_path)
	labels = get_wav_label(wav_files, label_files)
	word_num_map = get_vocabulary(labels)
	return word_num_map

map_ = word2vec()

map_ = {a:b for b,a in map_.items()}

with open("./word_num_map.txt", 'wb') as f:
	pickle.dump(map_, f)
	f.close()