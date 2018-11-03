from han import HAN
from preprocess import Preprocessor
from minibatch import BatchFeeder
import  tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

EPOCH = 40

if __name__ == "__main__":
	bf = BatchFeeder('train', 64, 15, 50)

	tf.reset_default_graph()

	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
	
	with tf.device('/device:GPU:0'):
		model = HAN(vocab_size=len(bf.vocab_encode_dict.keys()) + 1, embedding_size=200, classes=10,
			word_cell=tf.nn.rnn_cell.GRUCell(50, name='word-gru'),
			sentence_cell=tf.nn.rnn_cell.GRUCell(50, name='sentence-gru'), word_context_size=100,
			sentence_context_size=100)

	saver = tf.train.Saver()

	with tf.Session(config=config) as sess:
		writer = tf.summary.FileWriter('./han_graph', graph=tf.get_default_graph())
		sess.run(tf.global_variables_initializer())
		for i in range(EPOCH):
			for encoded_data, document_length_mask, sentence_length_mask, labels in bf:
				feed_dict = {
					model.inputs: encoded_data,
					model.word_length: sentence_length_mask,
					model.sentence_length: document_length_mask,
					model.labels: labels
				}
				global_step, accuracy, summary, _ = sess.run([model.global_steps, model.accuracy, model.summary, model.train], feed_dict=feed_dict)
				print(global_step)
				print('Epoch {}/accuracy: {}'.format(i, accuracy))
				writer.add_summary(summary)
			saver.save(sess, './checkpoints/han', global_step=model.global_steps)
	writer.close()
