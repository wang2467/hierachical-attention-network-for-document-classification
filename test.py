from han import HAN
from minibatch import BatchFeeder
import tensorflow as tf
import os

def main():
	bf = BatchFeeder('test', 64, 50, 100)
	
	print(len(bf.vocab_encode_dict.keys()))
	
	tf.reset_default_graph()
	config = tf.ConfigProto(allow_soft_placement=True)
	
	ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))

	model = HAN(vocab_size=len(bf.vocab_encode_dict.keys()) + 1, embedding_size=200, classes=10,
		word_cell=tf.nn.rnn_cell.GRUCell(50, name='word-gru'),
		sentence_cell=tf.nn.rnn_cell.GRUCell(50, name='sentence-gru'), word_context_size=100,
		sentence_context_size=100)
	saver = tf.train.Saver()
	
	prediction_total = []
	labels_total = []
	with tf.Session(config=config) as sess:
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
		
		total_accuracy = 0
		total_count = 0
		for encoded_data, document_length_mask, sentence_length_mask, labels in bf:
			feed_dict = {
					model.inputs: encoded_data,
					model.word_length: sentence_length_mask,
					model.sentence_length: document_length_mask,
					model.labels: labels
				}
			prediction, accuracy = sess.run([model.prediction, model.accuracy], feed_dict=feed_dict)
			print(accuracy)
			total_accuracy += accuracy
			total_count += 1
			print(total_accuracy)
			prediction_total.extend(prediction.tolist())
			labels_total.extend(labels.tolist())

		print('accuracy: {} %'.format(total_accuracy * 100 / total_count))
	
		with open('result_accuracy.txt', 'w') as myFile:
			myFile.write('accuracy: {} %\n'.format(total_accuracy * 100 / total_count)) 
		
		with open('./data/result.txt', 'w') as myFile:
			for idx, res in enumerate(prediction_total):
				myFile.write('{}---{}\n'.format(res, labels_total[idx]))			

if __name__ == "__main__":
	main()
