from minibatch import BatchFeeder
import numpy as np
import pickle
import time

def main():
	bf = BatchFeeder('train',64, 50, 100)
	
	with open('./data/vocab_encode_dict.pkl', 'rb') as myFile:
		encode_dict = pickle.load(myFile)
	
	reverse_encode_dict = {value: key for key, value in encode_dict.items()}
	
	for idx, (encoded_data, document_length_mask, sentence_length_mask, labels) in enumerate(bf):
		if 197 <= idx <= 400:
			for doc_num, doc in enumerate(encoded_data):
				for i in range(document_length_mask[doc_num]):
					sent = doc[i]
					valid_sent_length = sentence_length_mask[doc_num, i]	
					print(' '.join([reverse_encode_dict[sent[k]] for k in range(valid_sent_length)]))
					time.sleep(5)

if __name__ == "__main__":
	main()
