import pickle
import numpy as np

def main():
	data = np.load('./data/encoded_test_50_100.npy')
	document_length_mask = np.load('./data/document_length_mask_test_50_100.npy')
	sentence_length_mask = np.load('./data/sentence_length_mask_test_50_100.npy')
	with open('./data/vocab_encode_dict.pkl', 'rb') as myFile:
		encode_dict = pickle.load(myFile)
	reverse_encode_dict = {value: key for key, value in encode_dict.items()}

	for i, doc in enumerate(data):
		for j, sent in enumerate(doc):
			sent_decoded = ' '.join([reverse_encode_dict[word] for word in sent])
			print(sent_decoded)
			print(sentence_length_mask[i, j])
		print(document_length_mask[i])
		if i == 1:
			break

if __name__ == "__main__":
	main()
