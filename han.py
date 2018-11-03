import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import pickle
import gensim

class HAN:
	def __init__(self, vocab_size, embedding_size, classes, word_context_size, sentence_context_size, word_cell, sentence_cell):
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.classes = classes
		self.word_context_size = word_context_size
		self.sentence_context_size = sentence_context_size
		self.word_cell = word_cell
		self.sentence_cell = sentence_cell

		with open('./data/vocab_encode_dict.pkl', 'rb') as myFile:
			dct = pickle.load(myFile)

		reverse_dct = {value: key for key, value in sorted(dct.items(), key=lambda x: x[1])}

		model = gensim.models.Word2Vec.load('./data/w2v.model')

		self.w2v = np.random.uniform(-np.sqrt(1 / 200), np.sqrt(1 / 200), (len(reverse_dct), 200))

		for key, value in reverse_dct.items():
			if value != 'UNK':
				self.w2v[key, :] = model.wv.get_vector(value)

		with tf.variable_scope('han') as scope:
			# document * sentence * word
			self.inputs = tf.placeholder(name='inputs', shape=(None, None, None), dtype=tf.int32)

			# document * sentence
			self.word_length = tf.placeholder(name='word_length', shape=(None, None), dtype=tf.int32)

			# document
			self.sentence_length = tf.placeholder(name='sentence_length', shape=(None, ), dtype=tf.int32)

			# A * (B, C) = tf.unstack (A, B, C)
			(self.document_size, self.sentence_size, self.word_size) = tf.unstack(tf.shape(self.inputs))

			# labels
			self.labels = tf.placeholder(name='labels', shape=(None, ), dtype=tf.int32)

			# global steps
			self.global_steps = tf.get_variable(name='global_steps', initializer=tf.constant(0), trainable=False, dtype=tf.int32)

			# embedding layer
			with tf.variable_scope('embedding') as scope:
				self.embedding_matrix = tf.get_variable(name='embedding_matrix', initializer=tf.constant(self.w2v, dtype=tf.float32), dtype=tf.float32)
				self.inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.inputs)

			# word level inputs
			word_level_inputs = tf.reshape(self.inputs_embedded, [self.document_size * self.sentence_size, self.word_size, self.embedding_size], name='word_level_inputs')
			word_level_length = tf.reshape(self.word_length, [self.document_size * self.sentence_size], name='word_level_length')

			# word level bidirectional-gru encoder
			with tf.variable_scope('word') as scope:
				with tf.variable_scope('bidirectional-gru') as scope:
					word_level_encoder_outputs = self.bidirectional_gru(scope, word_cell, word_cell, word_level_inputs, word_level_length)
					self.word_level_encoder_outputs = word_level_encoder_outputs

				# word level attention (here we assume the word_context_size is the same as the word level encoder hidden dimension)
				with tf.variable_scope('attention') as scope:
					word_level_attention_outputs = self.attention_layer(scope, word_level_encoder_outputs, self.word_context_size)
					self.word_level_attention_outputs = word_level_attention_outputs

				# word level drop-out layer
				with tf.variable_scope('dropout') as scope:
					word_level_outputs = layers.dropout(word_level_attention_outputs, keep_prob=0.5, is_training=True)

			# sentence level inputs (here we assume the sentence_context_size is the same as the sentence level encoder hidden dimension)
			sentence_level_inputs = tf.reshape(word_level_outputs, [self.document_size, self.sentence_size, self.word_context_size], name='sentence_level_inputs')
			self.sentence_level_inputs = sentence_level_inputs

			# sentence level bidirectional-gru encoder
			with tf.variable_scope('sentence') as scope:
				with tf.variable_scope('bidirectional-gru') as scope:
					sentence_level_encoder_outputs = self.bidirectional_gru(scope, sentence_cell, sentence_cell, sentence_level_inputs, self.sentence_length)
					self.sentence_level_encoder_outputs = sentence_level_encoder_outputs

				# sentence level attention
				with tf.variable_scope('attention') as scope:
					sentence_level_attention_outputs = self.attention_layer(scope, sentence_level_encoder_outputs, self.sentence_context_size)
					self.sentence_level_attention_outputs = sentence_level_attention_outputs

				# sentence level drop-out layer
				with tf.variable_scope('dropout') as scope:
					sentence_level_outputs = layers.dropout(sentence_level_attention_outputs, keep_prob=0.5, is_training=True)
					self.sentence_level_outputs = sentence_level_outputs

			# final dense layer
			with tf.variable_scope('classification') as scope:
				self.logits = layers.fully_connected(sentence_level_outputs, self.classes,  activation_fn=None)

			self.prediction = tf.argmax(self.logits, axis=-1)

		with tf.variable_scope('train') as scope:
			# compute cross entropy
			self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)

			# sum all cross entropy to get the total loss
			self.loss = tf.reduce_mean(self.cross_entropy)
			tf.summary.scalar(name='loss', tensor=self.loss)

			# compute the accuracy of the prediction across the batch size inputs
			self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=self.logits, targets=self.labels, k=1), tf.float32))
			tf.summary.scalar(name='accuracy', tensor=self.accuracy)
			
			# get all trainable variables(except global_step because trainable is set to False)
			trainable_variables = tf.trainable_variables()

			# compute gradients w.r.t all trainable variables with max clip of 5.0(useful for bptt)
			gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_variables), 5.0)

			# set learning rate to 0.0001
			#optimizer = tf.train.AdamOptimizer(0.0001)
			optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
			self.train = optimizer.apply_gradients(zip(gradients, trainable_variables), name='train', global_step=self.global_steps)
			self.summary = tf.summary.merge_all()


	def bidirectional_gru(self, scope, fw_cell, bw_cell, inputs_embedded, input_length):
		'''
		:param scope: scope
		:param fw_cell: GRU cell
		:param bw_cell: GRU cell
		:param inputs_embedded: batch_size(document_size * sentence_size) * word_size(the number of words in each sentence) * embedding_size
		:param input_length: the number of sentences in all documents (document_size * sentence_size)
		:return: encoded representation for each word
		'''
		with tf.variable_scope(scope or 'bidirecitonal-rnn') as scope:
			((fw_outputs, bw_outputs), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=inputs_embedded, sequence_length=input_length, dtype=tf.float32, swap_memory=True, scope=scope)
			outputs = tf.concat([fw_outputs, bw_outputs], axis=2)

		return outputs

	def attention_layer(self, scope, inputs, context_size):
		'''
		:param scope: scope
		:param inputs: batch_size(document * sentence) * word_length(words in each sentence) * bi-gru_hidden_dimension
		:param context_size: context vector size (usually is consistent with bi-gru_hidden_dimension)
		:return: sentence representation batch_size * bi-gru_hidden_dimension
		'''
		with tf.variable_scope(scope) as scope:
			u_w = tf.get_variable(name='attention_context_vector', shape=[context_size], initializer=layers.xavier_initializer(), dtype=tf.float32)
			u_it = layers.fully_connected(inputs, context_size, activation_fn=tf.tanh, scope=scope)

			# dot product can be interpreted as element-wise multiplication with addition on that axis
			# [[1 2 3] element-wise multiply [1 2 3].T => [[1 4 9]     add across column => [13 32]
			#  [4 5 6]]                                    [4 10 18]]
			alpha_it = tf.nn.softmax(tf.reduce_sum(tf.multiply(u_it, u_w), axis=2, keepdims=True), axis=1, name='softmax')
			s_it = tf.multiply(alpha_it, inputs)
			s_t = tf.reduce_sum(s_it, axis=1)

		return s_t

if __name__ == "__main__":
	inputs = tf.placeholder(name='inputs', shape=(5, 2, 3), dtype=tf.float32)
	output_size = 10

	u_w = tf.get_variable(name='attention_context_vector', shape=[output_size], initializer=layers.xavier_initializer(),
	                      dtype=tf.float32)
	u_it = layers.fully_connected(inputs, output_size, activation_fn=tf.tanh)
	alpha_it = tf.nn.softmax(tf.reduce_sum(tf.multiply(u_it, u_w), axis=2, keepdims=True), axis=1)
	s_it = tf.multiply(alpha_it, inputs)
	s_t = tf.reduce_sum(s_it, axis=1)

	# tf.reset_default_graph()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		feed_dict = {inputs: np.arange(30).reshape((5, 2, 3)).astype(np.float32)}
		print(sess.run(tf.shape(alpha_it), feed_dict=feed_dict))
		print(sess.run(tf.shape(u_w), feed_dict=feed_dict))
		print(sess.run(tf.shape(tf.transpose(u_it)), feed_dict=feed_dict))
		print(sess.run(tf.shape(s_it), feed_dict=feed_dict))
		print(sess.run(tf.shape(s_t), feed_dict=feed_dict))
