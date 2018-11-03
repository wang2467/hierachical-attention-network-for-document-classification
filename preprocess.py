from collections import OrderedDict
from bs4 import BeautifulSoup
import numpy as np
import pickle
import nltk
import re
import gensim

class Preprocessor:
	def __init__(self, fname, mode):
		self.mode = mode
		self.vocab_freq_dict = {}
		self.vocab_encode_dict = {}
		self.cleaned_data = []
		self._load_data(fname)

	def buildDictionary(self):
		model = gensim.models.Word2Vec.load('./data/w2v.model')

		for idx, word in enumerate(model.wv.vocab):
			self.vocab_encode_dict[word] = idx + 1
		self.vocab_encode_dict['UNK'] = 0

		with open('./data/vocab_encode_dict.pkl', 'wb') as myFile:
			pickle.dump(self.vocab_encode_dict, myFile)		

	def buildDictionary_obsolete(self):
		print('Starting cleaning data')
		self._clean_data()
		print('End cleaning data')
		print('-'*20)

		print('Start filtering dict')
		for word in list(self.vocab_freq_dict.keys()):
			if self.vocab_freq_dict[word] <= 5:
				del self.vocab_freq_dict[word]
		print('End filtering dict')
		print('-'*20)

		self.vocab_freq_dict = sorted(self.vocab_freq_dict.items(), key=lambda x: (x[1], x[0]), reverse=True)

		print('Start building encoding dict')
		for idx, word in enumerate(list(self.vocab_freq_dict)):
			self.vocab_encode_dict[word[0]] = idx + 1
		print('End building encoding dict')
		print('-'*20)

		self.vocab_encode_dict['UNK'] = 0

		print('Start dumping')
		with open('./data/vocab_encode_dict.pkl', 'wb') as myFile:
			pickle.dump(self.vocab_encode_dict, myFile)

		self.vocab_freq_dict = OrderedDict(self.vocab_freq_dict)

		with open('./data/vocab_freq_dict.pkl', 'wb') as myFile:
			pickle.dump(self.vocab_freq_dict, myFile)
		print('End dumping')

	def encode_data(self, max_setence_size=50, max_word_size=100):
		if self.mode != 'build':
			with open('./data/vocab_encode_dict.pkl', 'rb') as myFile:
				self.vocab_encode_dict = pickle.load(myFile)

			self._clean_data()

		data = np.zeros((len(self.cleaned_data), max_setence_size, max_word_size), dtype=np.int32)
		word_length = np.zeros((len(self.cleaned_data), max_setence_size), dtype=np.int32)
		sentence_length = np.zeros(len(self.cleaned_data), dtype=np.int32)

		for doc_num, doc in enumerate(self.cleaned_data):
			print(doc_num)
			sentence_length[doc_num] = min(len(doc), max_setence_size)
			for sent_num, sent in enumerate(doc):
				if sent_num < max_setence_size:
					word_length[doc_num, sent_num] = min(len(sent), max_word_size)
					for word_num, word in enumerate(sent):
						if word_num < max_word_size:
							data[doc_num, sent_num, word_num] = self.vocab_encode_dict.get(word, 0)
	
		if self.mode == 'train':
			np.save('./data/encoded_train_{}_{}'.format(max_setence_size, max_word_size), data)
			np.save('./data/sentence_length_mask_{}_{}'.format(max_setence_size, max_word_size), word_length)
			np.save('./data/document_length_mask_{}_{}'.format(max_setence_size, max_word_size), sentence_length)
			np.save('./data/labels_{}_{}'.format(max_setence_size, max_word_size), np.array(self.labels))
		elif self.mode == 'test':
			np.save('./data/encoded_test_{}_{}'.format(max_setence_size, max_word_size), data)
			np.save('./data/sentence_length_mask_test_{}_{}'.format(max_setence_size, max_word_size), word_length)
			np.save('./data/document_length_mask_test_{}_{}'.format(max_setence_size, max_word_size), sentence_length)
			np.save('./data/labels_test_{}_{}'.format(max_setence_size, max_word_size), np.array(self.labels))
	
		return data, word_length, sentence_length

	def _load_data(self, fname):
		with open(fname, 'r') as myFile:
			data = myFile.read()

		data = re.split(r'\n', data)[:-1]

		self.raw_data = []
		self.labels = []

		for c, i in enumerate(data):
			if i[-2].isdigit():
				self.raw_data.append(i[:-5])
				self.labels.append(int(i[-2:])-1)
			else:
				self.raw_data.append(i[:-4])
				self.labels.append(int(i[-1:])-1)

	def _clean_data(self):
		cleaned_doc_list = []
		for doc in self.raw_data:
			cleaned_sent_list = []
			sent_list = self._tokenize_sentence(doc)
			for sent in sent_list:
				sent = self._remove_special_characters(sent)
				word_list = self._tokenize_word(sent)
				cleaned_sent_list.append(word_list)
			cleaned_doc_list.append(cleaned_sent_list)
		self.cleaned_data = cleaned_doc_list


	def _tokenize_sentence(self, text):
		text = text.lower()
		text = self._remove_html(text)
		sent_list = nltk.sent_tokenize(text)
		return sent_list

	def _tokenize_word(self, sent):
		tokens = nltk.word_tokenize(sent.lower())
		result = []
		for i in tokens:
			result.append(i)
			if self.mode == 'build':
				if i not in self.vocab_freq_dict:
					self.vocab_freq_dict[i] = 0
				self.vocab_freq_dict[i] += 1
		return result

	def _remove_html(self, text):
		return BeautifulSoup(text, 'html.parser').get_text()
	def _remove_special_characters(self, text):
		text = re.sub(r'[^a-zA-Z\s]', '', text)
		return text

if __name__ == '__main__':
	p = Preprocessor('./data/test.txt', 'test')
	#p.buildDictionary()
	data, sentence_mask, document_mask = p.encode_data(15, 50)
	print(data[5])
	print(sentence_mask[5])
	print(document_mask[5])
