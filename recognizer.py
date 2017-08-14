from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model import speech2text
from scipy.io import wavfile as wavio
import pickle
import librosa
import tensorflow as tf 
import numpy as np

with open("./word_num_map.txt", 'rb') as f:
	word_map =  pickle.load(f)
	f.close()

def recognize(wav_file):
	sr, wav = wavio.read(wav_file)
	wav = wav.astype('float32') / 32767
	mfcc = np.transpose(np.expand_dims(librosa.feature.mfcc(wav, sr), axis=0), [0, 2, 1])
	tmp = tf.placeholder(dtype=tf.float32, shape=[1, None, 20])
	sequence_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(tmp, reduction_indices=2), 0.), tf.int32), reduction_indices=1)
	inference = speech2text(X=tmp)
	saver = tf.train.Saver()

	with tf.Session() as sess:
		saver.restore(sess, tf.train.latest_checkpoint("./model/"))
		decoded = tf.transpose(inference, perm=[1, 0, 2])
		decoded, _ = tf.nn.ctc_beam_search_decoder(decoded, sequence_len, merge_repeated=False)
		predict = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values) + 1
		result = sess.run(predict, feed_dict={tmp: mfcc})
	print(result)
# issue: [SparseTensorValue(indices=array([], shape=(0, 2), dtype=int64), values=array([], dtype=int64), dense_shape=array([1, 0], dtype=int64))]
recognize("./test.wav")

