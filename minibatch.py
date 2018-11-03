import numpy as np
import pickle

class BatchFeeder:
	def __init__(self, mode, batch_szie, max_sentence_length, max_word_length):
		self.batch_size = batch_szie
		with open('./data/vocab_encode_dict.pkl', 'rb') as myFile:
			self.vocab_encode_dict = pickle.load(myFile)
		if mode == 'train':
			self.data = np.load('./data/encoded_train_{}_{}.npy'.format(max_sentence_length, max_word_length))
			self.document_length_mask = np.load('./data/document_length_mask_{}_{}.npy'.format(max_sentence_length, max_word_length))
			self.sentence_length_mask = np.load('./data/sentence_length_mask_{}_{}.npy'.format(max_sentence_length, max_word_length))
			self.labels = np.load('./data/labels_{}_{}.npy'.format(max_sentence_length, max_word_length))
		elif mode == 'test':
			self.data = np.load('./data/encoded_test_{}_{}.npy'.format(max_sentence_length, max_word_length))
			self.document_length_mask = np.load('./data/document_length_mask_test_{}_{}.npy'.format(max_sentence_length, max_word_length))
			self.sentence_length_mask = np.load('./data/sentence_length_mask_test_{}_{}.npy'.format(max_sentence_length, max_word_length))
			self.labels = np.load('./data/labels_test_{}_{}.npy'.format(max_sentence_length, max_word_length))
			
		self.index_sequence = np.random.permutation(len(self.data))
		self.curr_idx = 0

	def __next__(self):
		if self.curr_idx < self.data.shape[0]:
			encoded_data = self.data[self.index_sequence[self.curr_idx:min(self.data.shape[0], self.curr_idx + self.batch_size)], :, :]
			sentence_length_mask = self.sentence_length_mask[self.index_sequence[self.curr_idx:min(self.data.shape[0], self.curr_idx + self.batch_size)], :]
			document_length_mask = self.document_length_mask[self.index_sequence[self.curr_idx:min(self.data.shape[0], self.curr_idx + self.batch_size)]]
			labels = self.labels[self.index_sequence[self.curr_idx:min(self.data.shape[0], self.curr_idx + self.batch_size)]]
			self.curr_idx += self.batch_size

			return encoded_data, document_length_mask, sentence_length_mask, labels
		else:
			self.index_sequence = np.random.permutation(len(self.data))
			self.curr_idx = 0
			raise StopIteration

	def __iter__(self):
		return self

if __name__ == "__main__":
	bf = BatchFeeder('test', 64, 50, 100)
	for a, b, c ,d in bf:
		print(a.shape)
		print(b.shape)
		print(c.shape)
		print(d.shape)
