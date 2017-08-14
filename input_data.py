from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import librosa
import scipy.io.wavfile as wavio
import numpy as np
import os
from collections import Counter
from pylsy import pylsytable

def get_wav_files(wav_path):
	print("Loading `.wav` file list......")
	wav_files = []
	for (dirpath, dirname, filename) in os.walk(wav_path):
		for _filename in filename:
			if _filename.endswith('.wav') or _filename.endswith('.WAV'):
				filename_path = os.sep.join([dirpath, _filename])
				wav_files.append(filename_path)
	print("Done.")
	return wav_files

def get_wav_label(wav_files, label_files):
	print("Loading labels......")
	label_dict = {}
	with open(label_files, 'r', encoding='utf-8') as f:
		for label in f:
			label = label.strip('\n')
			label_id = label.split(' ', 1)[0]
			label_text = label.split(' ', 1)[1]
			label_dict[label_id] = label_text
	labels = []
	new_wav_files = []
	for wav_file in wav_files:
		wav_id = os.path.basename(wav_file).split('.')[0]
		if wav_id in label_dict:
			labels.append(label_dict[wav_id])
			new_wav_files.append(wav_file)
	print("Done.")
	return new_wav_files, labels

def get_vocabulary(labels):
	print("Loading vocabulary......")
	all_words = []
	for label in labels:
	    all_words += [word for word in label]
	counter = Counter(all_words)
	count_pairs = sorted(counter.items(), key=lambda x: -x[1])
	words, _ = zip(*count_pairs)
	words_size = len(words)

	word_num_map = dict(zip(words, range(len(words))))
	to_num = lambda word: word_num_map.get(word, len(words))
	labels_vector = [list(map(to_num, label)) for label in labels]
	label_max_len = np.max([len(label) for label in labels_vector])
	print("Done.")
	return words_size, label_max_len, labels_vector

def get_mfcc(wav_files):
	print("Loading `.wav` files......")
	wav_max_len = 0
	for wav_ in wav_files:
		sr, wav = wavio.read(wav_)
		wav = wav.astype('float32') / 32767
		mfcc = np.transpose(librosa.feature.mfcc(wav, sr), [1,0])
		if len(mfcc) > wav_max_len:
			wav_max_len = len(mfcc)
	print("Done.")
	return mfcc, wav_max_len

wav_path = './data/wav/train'
label_files = './data/doc/trans/train.word.txt'
wav_files = get_wav_files(wav_path)
wav_files, labels = get_wav_label(wav_files, label_files)
words_size, label_max_len, labels_vector = get_vocabulary(labels)
#mfcc, wav_max_len = get_mfcc(wav_files)

attributes = ["AttrName", "AttrValue"]
table = pylsytable(attributes)
AttrName = [
	"Longest Speech",
	"Samples Number",
	"Vocabulary Size",
	"Longest Sentence Word Number"]
AttrValue = [
	wav_max_len,
	len(wav_files),
	words_size,
	label_max_len]
table.add_data("AttrName", AttrName)
table.add_data("AttrValue", AttrValue)
print(table)

pointer = 0
def next_batch(batch_size):
	global pointer
	batches_wavs = []
	batches_labels = []
	for i in range(batch_size):
		sr, wav = wavio.read(wav_files[pointer])
		wav = wav.astype('float32') / 32767
		mfcc = np.transpose(librosa.feature.mfcc(wav, sr), [1, 0])
		batches_wavs.append(mfcc.tolist())
		batches_labels.append(labels_vector[pointer])
		pointer += 1
	for mfcc in batches_wavs:
		while len(mfcc) < wav_max_len:
			mfcc.append([0] * 20)
	for label in batches_labels:
		while len(label) < label_max_len:
			label.append(0)
	return batches_wavs, batches_labels