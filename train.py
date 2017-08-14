from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model import *
import tensorflow as tf
import datetime

n_batch = len(wav_files) // batch_size
class MaxPropOptimizer(tf.train.Optimizer):
	def __init__(self, learning_rate=0.001, beta2=0.999, use_locking=False, name="MaxProp"):
		super(MaxPropOptimizer, self).__init__(use_locking, name)
		self._lr = learning_rate
		self._beta2 = beta2
		self._lr_t = None
		self._beta2_t = None
	def _prepare(self):
		self._lr_t = tf.convert_to_tensor(self._lr, name="learning_rate")
		self._beta2_t = tf.convert_to_tensor(self._beta2, name="beta2")
	def _create_slots(self, var_list):
		for v in var_list:
			self._zeros_slot(v, "m", self._name)
	def _apply_dense(self, grad, var):
		lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
		beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
		if var.dtype.base_dtype == tf.float16:
			eps = 1e-7
		else:
			eps = 1e-8
		m = self.get_slot(var, "m")
		m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
		g_t = grad / m_t
		var_update = tf.assign_sub(var, lr_t * g_t)
		return tf.group(*[var_update, m_t])
	def _apply_sparse(self, grad, var):
		return self._apply_dense(grad, var)

def train():
	sess = tf.Session()
	logit = speech2text(X=X)

	# CTC loss
	indices = tf.where(tf.not_equal(tf.cast(Y, tf.float32), 0.))
	target = tf.SparseTensor(indices=indices, values=tf.gather_nd(Y, indices) - 1, dense_shape=tf.cast(tf.shape(Y), tf.int64))
	loss = tf.nn.ctc_loss(target, logit, sequence_len, time_major=False)
	avg_loss = tf.reduce_mean(loss)
	# optimizer
	lr = tf.Variable(0.001, dtype=tf.float32, trainable=False)
	optimizer = MaxPropOptimizer(learning_rate=lr, beta2=0.99)
	var_list = [t for t in tf.trainable_variables()]
	gradient = optimizer.compute_gradients(loss, var_list=var_list)
	optimizer_op = optimizer.apply_gradients(gradient)

	tf.summary.scalar("AverageLoss", avg_loss)
	logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
	writer = tf.summary.FileWriter(logdir, sess.graph)
	saver = tf.train.Saver(tf.global_variables())
	merged = tf.summary.merge_all()
	model_dir = "./model/"
	if os.path.isdir(model_dir):
		pass
	else:
		os.makedirs(model_dir)
	with tf.name_scope("Train"):
		print("Start Training......")
		sess.run(tf.global_variables_initializer())
		sess.run(tf.assign(lr, 0.001))
		global pointer
		pointer = 0
		for batch in range(n_batch):
			batches_wavs, batches_labels = next_batch(batch_size)
			summary, _loss, train_loss, _ = sess.run([merged, avg_loss, loss, optimizer_op], feed_dict={X: batches_wavs, Y: batches_labels})
			time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
			print(" "+time+":"+" Batch:{} Average Loss:{:.4f}".format(batch, _loss))
			writer.add_summary(summary, batch)
			saver.save(sess, model_dir, global_step=batch)

	sess.close()

train()